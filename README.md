# layout-classify

[synth-classify](../synth-classify/README.md) measured local text classifiers on short messages.
This repo asks the same questions of document pages, using 9 types from
RVL-CDIP-N, modern pages labelled with RVL-CDIP's types, plus pages that fit none of them.
So far it holds the data, the OCR cache, two classifiers on the OCR text and one on the page image.

## Aim

Four questions, measured on one laptop:

1. **How much work can it do without a person checking?** The classifier gives each page a type and
   a score. Above a cut-off, the type is used and nobody reads the page.
   *Coverage* is the share of test pages above the cut-off.
   *Error* is the share of those whose type is wrong.
   We report coverage at 5% error, with 90 and 288 hand-labelled pages.
2. **Does the page image beat OCR text?** Same pages, same tables; one classifier reads the text,
   another reads the image.
3. **What do pages that fit no type cost?** Production must send these to a person.
   Coverage and error are reported with and without them.
4. **How many seconds per page?**

## Data

All pages come from RVL-CDIP-N, in `~/data/rvl-cdip-n/RVL-CDIP-N/`: 1002 modern pages from the web
and DocumentCloud, labelled with RVL-CDIP's types. It has 12 of RVL-CDIP's 16 types. The 9 types are
`CLASSES` in `classes.py`: letter, memo, form, questionnaire, budget, invoice, scientific
publication, email and news article. They include three near-pairs: letter / memo, form /
questionnaire and budget / invoice. A reject page comes from one of the other 7 RVL-CDIP types,
`REJECT_CLASSES`; RVL-CDIP-N has three of them, handwritten, specification and resume. A reject
page's label is `none of these`, and its `rvl_label` keeps the RVL-CDIP type.

Each type's pages are shuffled with seed 0. A type in `CLASSES` puts half its pages, rounded down,
in train, a fifth, rounded down, in dev, and the rest in test. A reject type puts none in train, 20
in dev and 34 or 33 in test, handwritten taking the extra one; its other pages are not used.

| Type                   | Train | Dev | Test | Unused |
|------------------------|-------|-----|------|--------|
| letter                 | 76    | 30  | 46   | 0      |
| memo                   | 23    | 9   | 15   | 0      |
| form                   | 35    | 14  | 21   | 0      |
| questionnaire          | 19    | 7   | 13   | 0      |
| budget                 | 29    | 11  | 18   | 0      |
| invoice                | 28    | 11  | 18   | 0      |
| scientific publication | 19    | 7   | 13   | 0      |
| email                  | 16    | 6   | 11   | 0      |
| news article           | 43    | 17  | 26   | 0      |
| handwritten (reject)   | 0     | 20  | 34   | 122    |
| specification (reject) | 0     | 20  | 33   | 8      |
| resume (reject)        | 0     | 20  | 33   | 131    |
| **Total**              | 288   | 172 | 281  | 261    |

Test has 181 in-class pages and 100 reject pages; dev has 112 and 60.

data/selection.json records the file names in each split and type, and the unused reject pages
under `unused`, so the set can be rebuilt. A page's `rvl_index` is its RVL-CDIP-N file name. A page
that fails to decode is logged, listed in data/selection.json, and replaced by the next page in the
same seeded order. None failed.

## Files

- `classes.py` — `CLASSES`, the 9 types with a one-sentence definition each; `REJECT_CLASSES`, the
  other 7 RVL-CDIP types; `REJECT_LABEL`, `"none of these"`. `test_classes.py` checks that `CLASSES`
  and `REJECT_CLASSES` together are the 16 label names of `aharley/rvl_cdip`.
- `data.py` — draws the three splits from RVL-CDIP-N and writes `data/<split>/<id>.png`,
  `data/manifest.jsonl` (fields `id`, `split`, `label`, `rvl_label`, `rvl_index`) and
  `data/selection.json`.
- `ocr.py` — runs Tesseract once on every page in the manifest. It writes `ocr/<id>.json`:
  `{"text", "words": [{"text", "x0", "y0", "x1", "y1", "conf"}]}`, with boxes in pixels of the
  page image. It skips pages already cached. Each run that OCRs a page appends a block to
  `results/ocr-time.txt`, headed by the date and the splits that had pages run.
- `train.py` — fits a classifier on the OCR text of train pages and saves it to `models/<name>/`.
  `--tfidf` is TF-IDF + logistic regression; `--embed` is logistic regression on frozen
  `dmis-lab/UltraBERT-Embed-150M` embeddings, at synth-classify's pinned revision, with a task
  instruction before the page text and the first 512 tokens kept. `--train real` is all 288 train
  pages; `--train small` is 10 per class, 90 in all, drawn with seed 0, the ids in
  data/selection.json under `small`. Neither holds a reject page. Every classifier uses
  `class_weight="balanced"`. `--dit` reads the page image instead: logistic regression on frozen
  `microsoft/dit-base` embeddings, at a pinned revision, each the mean of the last hidden state over
  the 196 patches of the page resized to 224 × 224, standardised by a `StandardScaler` fitted on
  the train embeddings. Names are `tfidf-real`, `tfidf-small`,
  `ultrabert-embed-real`, `ultrabert-embed-small`, `dit-real` and `dit-small`.
- `score.py` — scores a model on every page of dev or test, reject pages included. A model
  predicts one of `CLASSES`, so a reject page is always wrong. A `dit-` model reads
  `data/<split>/<id>.png`; its seconds leave out the image read. `--dump` writes each page's
  prediction and top probability for `confidence.py`.
- `confidence.py` — adapted from synth-classify's. It picks the cut-offs on every dev page, then
  reports on test, at each cut-off, coverage and error on in-class pages, the reject leak, and
  coverage and error for a stream with 10% reject pages and for all test pages.
- `data/` — the manifest and the selection are tracked. The page images are ignored by git and
  rebuilt by `data.py`.
- `ocr/` — the OCR cache, one JSON file per page.
- `models/` — trained classifiers, ignored by git and rebuilt by `train.py`.
- `results/` — every reported number, in synth-classify's layout without its class-set level:
  `results/<split>/<name>.txt` for score reports, `results/train/<name>.txt` for training time,
  `results/confidence/<name>.txt` for routing, with its inputs in `results/confidence/dumps/`, and
  `results/ocr-time.txt` for OCR time. `results/unweighted/` holds the train, score and
  confidence files from the run without class weights, `results/unscaled/` those of the `dit-`
  models without the scaler, and `results/rvl-cdip/` the same files from
  the RVL-CDIP runs in the appendix.

## Run

Tesseract must be installed (`brew install tesseract`).

```sh
uv sync
uv run pytest           # checks classes.py against aharley/rvl_cdip's label names
uv run python data.py   # about 12 s
uv run python ocr.py    # about 5 min

uv run python train.py --tfidf --train real          # also --train small; --embed for UltraBERT, --dit for DiT
for split in dev test; do
  uv run python score.py --model tfidf-real --split $split --dump
done
uv run python confidence.py --model tfidf-real
```

`--embed` and `--dit` training and scoring fetch the model from Hugging Face on first use.

## Results

All numbers are from `results/`. α is the error target. The cut-off is the lowest dev score at
which the error among dev pages at or above it is at most α. It is chosen on all 172 dev pages,
reject pages included, and applied unchanged to test. A reject page above it is an error. Seconds
are the median per page, one page per call, OCR excluded.

### Coverage and error at the 5% cut-off

*Reject leak* is the share of test reject pages that score at or above the cut-off. The *10% reject*
columns are for a stream of 90% in-class and 10% reject pages, computed from the in-class numbers
and the reject leak: coverage is 0.9 × in-class coverage + 0.1 × reject leak, and error is the
accepted reject pages plus the accepted in-class errors, over that coverage.

| Model                   | Hand labels | Coverage (in-class) | Error (in-class) | Reject leak | Coverage at 10% reject | Error at 10% reject |
|-------------------------|-------------|---------------------|------------------|-------------|------------------------|---------------------|
| `tfidf-small`           | 90          | 0.4420              | 0.0125           | 0.0200      | 0.3998                 | 0.0174              |
| `tfidf-real`            | 288         | 0.3536              | 0.0000           | 0.0000      | 0.3182                 | 0.0000              |
| `ultrabert-embed-small` | 90          | 0.0055              | 0.0000           | 0.0000      | 0.0050                 | 0.0000              |
| `ultrabert-embed-real`  | 288         | 0.2265              | 0.0000           | 0.0200      | 0.2059                 | 0.0097              |
| `dit-small`             | 90          | 0.0000              | -                | 0.0000      | 0.0000                 | -                   |
| `dit-real`              | 288         | 0.0000              | -                | 0.0000      | 0.0000                 | -                   |

From `results/confidence/<model>.txt`, cut-off `tau` under `alpha 0.05` in the test section. That
file also gives coverage and error over all 281 test pages, whose mix is 100 reject to 181
in-class. With the dev error replaced by its one-sided 90% Clopper-Pearson upper bound (`tau_cp`),
only `tfidf-small` accepts any page at α 0.05: 0.3690 of the 10% reject stream at 0.0027 error.
For `dit-small` and `dit-real` no dev score gives at most 5% error, so `tau` is `-` and they
accept no page.

### Accuracy

| Model                   | Hand labels | Test   | Test, in-class pages | Seconds per page |
|-------------------------|-------------|--------|----------------------|------------------|
| `tfidf-small`           | 90          | 0.4982 | 0.7735               | 0.000282         |
| `tfidf-real`            | 288         | 0.4911 | 0.7624               | 0.000255         |
| `ultrabert-embed-small` | 90          | 0.5231 | 0.8122               | 0.035            |
| `ultrabert-embed-real`  | 288         | 0.5587 | 0.8674               | 0.0351           |
| `dit-small`             | 90          | 0.4057 | 0.6298               | 0.0136           |
| `dit-real`              | 288         | 0.4235 | 0.6575               | 0.0137           |

Test is over all 281 pages, reject pages included. The in-class column is over the 181 pages whose
label is in `CLASSES` (line `accuracy … without reject pages`). All from `results/test/<model>.txt`.

### Hand labels against coverage

| Hand labels | Model         | Dev coverage | Coverage at 10% reject | Error at 10% reject |
|-------------|---------------|--------------|------------------------|---------------------|
| 90          | `tfidf-small` | 0.2965       | 0.3998                 | 0.0174              |
| 90          | `dit-small`   | 0.0000       | 0.0000                 | -                   |
| 288         | `tfidf-real`  | 0.2674       | 0.3182                 | 0.0000              |
| 288         | `dit-real`    | 0.0000       | 0.0000                 | -                   |

At each count, the text row is the text model with the higher dev coverage at α 0.05, over all 172
dev pages, and the `dit-` row is the image model; test was not used to choose either.

### Discussion

- `class_weight="balanced"` was set after the unweighted run showed `tfidf-real` below
  `tfidf-small` (`results/unweighted/`). `tfidf-real`'s in-class test accuracy is 0.6685 unweighted
  and 0.7624 balanced. That comparison was read on test, so this one choice was not fixed on dev
  alone. `--train small` already has 10 pages per class, so the weights leave the `-small` models
  unchanged. The `StandardScaler` in `--dit` was added after the unscaled run saturated at
  probability 1.0 (`results/unscaled/`).
- AUROC of score against correctness on all dev pages: `tfidf-small` 0.8552, `tfidf-real` 0.8349,
  `ultrabert-embed-small` 0.8315, `ultrabert-embed-real` 0.8554.
- UltraBERT-Embed has the higher in-class accuracy at both sizes, 0.8674 for
  `ultrabert-embed-real`, but its top probabilities on dev leave fewer pages above a 5% cut-off:
  0.2059 of the 10% reject stream for `ultrabert-embed-real` and 0.0050 for
  `ultrabert-embed-small`.
- Question 2: the OCR text beats the page image. At 10% reject, `tfidf-real` covers 0.3182 and
  `dit-real` 0.0000. With the scaler, `dit-real` still gives a wrong dev page a score of 1.0, so no
  cut-off reaches 5% dev error.
- AUROC on all dev pages for the image model: `dit-small` 0.6926, `dit-real` 0.7037.
- At the 5% cut-off no model lets through more than 2 of the 100 test reject pages.
- Seconds per page on test: TF-IDF about 0.0003, UltraBERT-Embed about 0.035, DiT about 0.014.

### Limits

- Test has 181 in-class pages, so a difference under 5 points between two models is noise.
- 7 of the 741 pages have no OCR words (`results/ocr-time.txt`). They are scored on empty text.
- RVL-CDIP-N's labels are used as published, not cleaned.
- Every number is from one run: one draw, one training run, one scoring run.

## Appendix: RVL-CDIP

The first runs used RVL-CDIP itself, whose pages are low-resolution 1990s scans from the tobacco
litigation archive and not the pages this work is meant to classify, so RVL-CDIP-N replaced it. They
had 10 types, scientific report included, and 6 reject types. The code that drew this set is
`data.py` at commit `e50680e`; its manifest and selection are in `data/` at that commit.

#### Data

The 10 types are `CLASSES` in `classes.py`: letter, memo, form, questionnaire, budget, invoice,
scientific report, scientific publication, email and news article. They include four near-pairs:
letter / memo, form / questionnaire, budget / invoice, and scientific report / scientific
publication. A reject page comes from one of the other 6 RVL-CDIP types, `REJECT_CLASSES`. Its label
is `none of these`, and its `rvl_label` keeps the RVL-CDIP type. Reject pages are spread evenly over
the 6 types. When the count does not divide by 6, the first types in `REJECT_CLASSES` take one more.

| Split    | Pages per class | Reject pages | Pages | Source                | Licence         |
|----------|-----------------|--------------|-------|-----------------------|-----------------|
| `train`  | 100             | 0            | 1000  | RVL-CDIP train        | research use    |
| `dev`    | 20              | 20           | 220   | RVL-CDIP validation   | research use    |
| `test`   | 40              | 40           | 440   | RVL-CDIP test         | research use    |
| `test-n` | as found        | 60           | 641   | RVL-CDIP-N            | CC BY 3.0       |

test-n is modern pages from the web and DocumentCloud, not 1990s scans, and scientific report is
absent from it. It keeps every RVL-CDIP-N page of a type in `CLASSES`: letter 152, memo 47, form 70,
questionnaire 39, budget 58, invoice 57, scientific publication 39, email 33, news article 86. Its
reject pages are 20 each of the three `REJECT_CLASSES` types RVL-CDIP-N has (handwritten,
specification, resume), drawn with seed 0. Its `rvl_index` is the RVL-CDIP-N file name.

The draw uses seed 0. data/selection.json records the RVL-CDIP indices drawn for each split and
type, so the set can be rebuilt. An index is a line number, from 0, in that split's label list
(`data/train.txt`, `data/val.txt` or `data/test.txt` in `aharley/rvl_cdip`). A page that fails to
decode is logged, listed in data/selection.json, and replaced by the next page of the same seeded
draw.

`aharley/rvl_cdip` is a loader script over one 38.8 GB tar.gz that holds all three splits. Current
`datasets` refuses loader scripts, and in streaming mode it would read the whole archive once per
split. So `data.py` makes the draw from the three label lists. It then streams the archive once,
keeps the drawn pages and writes nothing else to disk. It stops as soon as every drawn page has gone
by.

RVL-CDIP's labels have documented noise. They are used as published, not cleaned.

#### Results

All numbers are from `results/rvl-cdip/`. α is the error target. The cut-off is the lowest dev score
at which the error among dev pages at or above it is at most α. It is chosen on all 220 dev pages,
reject pages included, and applied unchanged to test and test-n. A reject page above it is an error.
Seconds are the median per page, one page per call, OCR excluded.

#### Coverage and error at the 5% cut-off

Test:

| Model                   | Hand labels | Coverage | Error  | Coverage no reject | Error no reject |
|-------------------------|-------------|----------|--------|--------------------|-----------------|
| `tfidf-small`           | 200         | 0.1636   | 0.0417 | 0.1800             | 0.0417          |
| `tfidf-real`            | 1000        | 0.2159   | 0.0211 | 0.2375             | 0.0211          |
| `ultrabert-embed-small` | 200         | 0.0159   | 0.0000 | 0.0175             | 0.0000          |
| `ultrabert-embed-real`  | 1000        | 0.2795   | 0.0894 | 0.3025             | 0.0744          |

Test-n:

| Model                   | Hand labels | Coverage | Error  | Coverage no reject | Error no reject |
|-------------------------|-------------|----------|--------|--------------------|-----------------|
| `tfidf-small`           | 200         | 0.0390   | 0.0000 | 0.0430             | 0.0000          |
| `tfidf-real`            | 1000        | 0.0920   | 0.0000 | 0.1015             | 0.0000          |
| `ultrabert-embed-small` | 200         | 0.0062   | 0.0000 | 0.0069             | 0.0000          |
| `ultrabert-embed-real`  | 1000        | 0.2699   | 0.0925 | 0.2909             | 0.0710          |

From `results/rvl-cdip/confidence/<model>.txt`, line `tau` under `alpha 0.05`. With the dev error
replaced by its one-sided 90% Clopper-Pearson upper bound (`tau_cp`), no model accepts any page at α
0.05.

#### Accuracy

| Model                   | Hand labels | Test   | Test-n | Test, in-class pages | Test-n, in-class pages | Seconds per page |
|-------------------------|-------------|--------|--------|----------------------|------------------------|------------------|
| `tfidf-small`           | 200         | 0.5500 | 0.6303 | 0.6050               | 0.6954                 | 0.000214         |
| `tfidf-real`            | 1000        | 0.5955 | 0.7395 | 0.6550               | 0.8158                 | 0.000256         |
| `ultrabert-embed-small` | 200         | 0.4955 | 0.5101 | 0.5450               | 0.5628                 | 0.0300           |
| `ultrabert-embed-real`  | 1000        | 0.6023 | 0.5725 | 0.6625               | 0.6317                 | 0.0303           |

Test and Test-n are over all pages, reject pages included. The in-class columns are over pages
whose label is in `CLASSES` (line `accuracy … without reject pages`). All from
`results/rvl-cdip/<split>/<model>.txt`. Seconds are on test.

#### Hand labels against coverage

| Hand labels | Best model             | Dev coverage | Test coverage | Test error |
|-------------|------------------------|--------------|---------------|------------|
| 200         | `tfidf-small`          | 0.1909       | 0.1636        | 0.0417     |
| 1000        | `ultrabert-embed-real` | 0.3000       | 0.2795        | 0.0894     |

Best is the model with the higher dev coverage at α 0.05; test was not used to choose it. All pages,
reject pages included.

RVL-CDIP's limits: 26 pages of train, dev and test and 5 of test-n have no OCR words
(`results/rvl-cdip/ocr-time.txt`); scientific report is absent from test-n.

## Licences

layout-classify is MIT; see `LICENSE`. RVL-CDIP is a subset of IIT-CDIP, from the Legacy Tobacco
Document Library, and is for research use. Its terms are at
<https://www.industrydocuments.ucsf.edu/help/copyright/>. RVL-CDIP-N is under Creative Commons
Attribution 3.0 Unported, per the `LICENSE.txt` it ships with. `microsoft/dit-base` is MIT, per
its model card.
