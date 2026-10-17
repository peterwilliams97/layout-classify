# layout-classify

[synth-classify](../synth-classify/README.md) measured local text classifiers on short messages.
This repo asks the same questions of scanned document pages, using 10 types from
[RVL-CDIP](https://huggingface.co/datasets/aharley/rvl_cdip) plus pages that fit none of them. So
far it holds the data and the OCR cache. It has no classifiers yet.

## Aim

Four questions, measured on one laptop:

1. **How much work can it do without a person checking?** Coverage at 5% error, as hand labels go
   from 0 to 200 to about 1,000.
2. **Does the page image or layout beat OCR text alone?** By how much, on the same tables.
3. **What does a document that fits no type cost?** Production must reject these. synth-classify's
   test set had none.
4. **Seconds per page** for each classifier. The device-side step has a time budget.

Coverage is the share of test pages a classifier labels on its own. Error is the share of those
pages whose label is wrong. Both are defined as in synth-classify.

## Data

The 10 types are `CLASSES` in `classes.py`: letter, memo, form, questionnaire, budget, invoice,
scientific report, scientific publication, email and news article. They include four near-pairs:
letter / memo, form / questionnaire, budget / invoice, and scientific report / scientific
publication. A reject page comes from one of the other 6 RVL-CDIP types, `REJECT_CLASSES`. Its label
is `none of these`, and its `rvl_label` keeps the RVL-CDIP type. Reject pages are spread evenly over
the 6 types. When the count does not divide by 6, the first types in `REJECT_CLASSES` take one more.

| Split   | Pages per class | Reject pages | Pages | RVL-CDIP source split |
|---------|-----------------|--------------|-------|-----------------------|
| `train` | 100             | 0            | 1000  | train                 |
| `dev`   | 20              | 20           | 220   | validation            |
| `test`  | 40              | 40           | 440   | test                  |

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
- `ocr.py` — runs Tesseract once on every page in the manifest. It writes `ocr/<id>.json`:
  `{"text", "words": [{"text", "x0", "y0", "x1", "y1", "conf"}]}`, with boxes in pixels of the
  page image. It skips pages already cached and writes the time it took to `results/ocr-time.txt`.
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
uv run python ocr.py
```

## Licences

layout-classify is MIT; see `LICENSE`. RVL-CDIP is a subset of IIT-CDIP, from the Legacy Tobacco
Document Library, and is for research use. Its terms are at
<https://www.industrydocuments.ucsf.edu/help/copyright/>.
