# layout-classify

[synth-classify](../synth-classify/README.md) measured local text classifiers on short messages.
This repo asks the same questions of scanned document pages, using 10 types from
[RVL-CDIP](https://huggingface.co/datasets/aharley/rvl_cdip) plus pages that fit none of them.
So far it holds the data and the OCR cache. It has no classifiers yet.

## Aim

Four questions, measured on one laptop:

1. **How much work can it do without a person checking?** The classifier gives each page a type and
   a score. Above a cut-off, the type is used and nobody reads the page.
   *Coverage* is the share of test pages above the cut-off.
   *Error* is the share of those whose type is wrong.
   We report coverage at 5% error, with 0, 200 and 1,000 hand-labelled pages.
2. **Does the page image beat OCR text?** Same pages, same tables; one classifier reads the  text,
   another reads the image.
3. **What do pages that fit no type cost?** Production must send these to a person.
   Coverage and error are reported with and without them.
4. **How many seconds per page?**

Coverage is the share of test pages a classifier labels on its own. Error is the share of those
pages whose label is wrong. Both are defined as in synth-classify.

## Data

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

## Files

- `classes.py` — `CLASSES`, the 10 types with a one-sentence definition each; `REJECT_CLASSES`, the
  other 6 RVL-CDIP types; `REJECT_LABEL`, `"none of these"`. `test_classes.py` checks that `CLASSES`
  and `REJECT_CLASSES` together are the 16 label names of `aharley/rvl_cdip`.
- `data.py` — draws the three splits and writes `data/<split>/<id>.png`, `data/manifest.jsonl`
  (fields `id`, `split`, `label`, `rvl_label`, `rvl_index`) and `data/selection.json`.
  `data.py test-n` adds test-n from RVL-CDIP-N in `~/data/rvl-cdip-n/RVL-CDIP-N/`; a plain
  `data.py` run rewrites both files without it, so `data.py test-n` runs again after one.
- `ocr.py` — runs Tesseract once on every page in the manifest. It writes `ocr/<id>.json`:
  `{"text", "words": [{"text", "x0", "y0", "x1", "y1", "conf"}]}`, with boxes in pixels of the
  page image. It skips pages already cached. Each run that OCRs a page appends a block to
  `results/ocr-time.txt`, headed by the date and the splits that had pages run.
- `confidence.py` — a copy of synth-classify's `confidence.py`, unchanged. It computes the
  dev-chosen confidence cut-off and the coverage and error at it. As copied, it imports
  `CLASS_SETS` from `classes.py`, which this repo does not define. It also exits on a label outside
  the class set, which includes `none of these`. A classifier task has to adapt it before it runs
  here.
- `data/` — the manifest and the selection are tracked. The page images are ignored by git and
  rebuilt by `data.py`.
- `ocr/` — the OCR cache, one JSON file per page.
- `results/` — every reported number, in synth-classify's layout, copied unchanged:
  `results/<set>/<split>/<name>.txt` for score reports, `results/<set>/train/<name>.txt` for
  training time, and `results/<set>/confidence/<name>.txt` with its `dumps/` for routing. So far it
  holds only `results/ocr-time.txt`.

## Run

Tesseract must be installed (`brew install tesseract`).

```sh
uv sync
uv run pytest           # checks classes.py against aharley/rvl_cdip's label names
uv run python data.py   # streams the 38.8 GB archive once; about 80 min at 8 MB/s
uv run python data.py test-n
uv run python ocr.py
```

## Licences

layout-classify is MIT; see `LICENSE`. RVL-CDIP is a subset of IIT-CDIP, from the Legacy Tobacco
Document Library, and is for research use. Its terms are at
<https://www.industrydocuments.ucsf.edu/help/copyright/>. RVL-CDIP-N is under Creative Commons
Attribution 3.0 Unported, per the `LICENSE.txt` it ships with.
