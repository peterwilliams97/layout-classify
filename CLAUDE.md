# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this
repository.

## What this is

A worked example, not a library. It is the sibling of `../synth-classify`, which measured local text
classifiers on Banking77. This repo asks the same questions of scanned pages from RVL-CDIP, plus a
reject class. The plan is `~/Downloads/layout-classify_plan.md` (version 1, 2026-10-17). It is not a
claude.ai artifact. It runs in six steps, one Claude Code prompt each. Step 1 has data and the OCR
cache. Steps 2–5 add the classifiers (TF-IDF + LR, UltraBERT-Embed-150M + LR, dit-base + LR,
layoutlmv3-base, gemma4:e4b on page images). Step 6 is the README.

The code is all Python. The repository layout, file header and `test.sh` rules in `~/CLAUDE.md` do
not apply here. Its Python / ML environment section does. Python files open with a module docstring
saying why the file exists, as in synth-classify.

## Commands

```sh
uv sync
uv run pytest                                   # the one test: classes.py vs aharley/rvl_cdip labels
uv run pytest test_classes.py::test_classes_cover_rvl_cdip_labels
uv run python data.py                           # ~60–75 min; streams the 38.8 GB archive once
uv run python ocr.py                            # Tesseract on every manifest page; resumable
```

Tesseract comes from Homebrew (`brew install tesseract`). Anything that reaches Hugging Face fails
inside the Claude Code sandbox with `OSStatus -26276`. That is macOS TLS trust lookup through
`truststore`. Run those commands, including the test, outside the sandbox.

There is no linter and no `test.sh`. As in synth-classify, the check on a change is the end-to-end
run.

## How the pieces connect

- **`classes.py` is the single source of the label space.** `CLASSES` holds the 10 types and their
  one-sentence definitions. The definitions are the only type description a prompt-based classifier
  gets. `REJECT_CLASSES` holds the other 6 RVL-CDIP types. `REJECT_LABEL = "none of these"`. Names
  are RVL-CDIP's own label strings, with spaces (`scientific report`), not identifiers.
- **`data.py` does not use `datasets`.** `aharley/rvl_cdip` is a loader script over one tar.gz, and
  current `datasets` refuses loader scripts. The draw is made from the repo's split lists
  `data/{train,val,test}.txt`, with one seeded `random.Random` per (split, type). One streamed
  `tarfile` pass then keeps the drawn pages. `rvl_index` is the line number, from 0, in that split
  list. Each draw takes `SPARE` extra pages in seeded order, which replace pages that fail to
  decode. The source splits differ (train / val / test), so our train, dev and test are disjoint.
  This differs from synth-classify, where dev was part of train.
- **Outputs.** `data/manifest.jsonl` has one row per page: `id`, `split`, `label`, `rvl_label`,
  `rvl_index`. `id` is `<split>-<rvl_index:06d>` and names both `data/<split>/<id>.png` and
  `ocr/<id>.json`. `data/selection.json` holds the seed, the drawn indices and `decode_failures`.
  Git ignores the page images (`data/*/`); the manifest and selection are tracked.
- **`ocr.py` is the only OCR path.** It reads the manifest, writes `ocr/<id>.json` (`text` plus
  word boxes in page pixels) and skips files already present. Every text classifier reads this
  cache and never re-OCRs. `results/ocr-time.txt` is the one OCR timing, reported apart from
  classifier seconds.
- **`confidence.py` is a verbatim copy of synth-classify's and does not run here yet.** It imports
  `CLASS_SETS` and exits on any label outside the class set, which includes `REJECT_LABEL`. The plan's
  reject rule is that a reject page above the cut-off counts as an error. Table 1 is reported with
  and without reject pages. A classifier step has to adapt `confidence.py` for that rule, in that
  one file: one confidence script, one scoring script.
- **`results/`** follows synth-classify's naming: `results/<set>/<split>/<name>.txt`, `train/`, and
  `confidence/` with `dumps/`. Every number reported needs a file there.

## Constraints from the plan

- Test is read once, after every cut-off and choice has been fixed on dev.
- Exclude `microsoft/dit-base-finetuned-rvlcdip`: it was trained on RVL-CDIP test pages.
- RVL-CDIP label noise is reported as a limit, not cleaned.
- Each step's prompt says: no branches, no commits, no push. Leave work uncommitted for Peter.
- Indra OCR (`prompts/2026-10-17-indra-ocr-pass.md`, a hand-off from the indra session) adds a
  second OCR path. The step 1 brief forbids that, so the hand-off is open pending Peter's decision.
