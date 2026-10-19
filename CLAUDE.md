# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this
repository.

## What this is

A worked example, not a library. It is the sibling of `../synth-classify`, which measured local text
classifiers on Banking77. This repo asks the same questions of document pages, plus a reject class.
The corpus must be native digital documents or good scans of them, rendered here at 200 dpi, split
train / validation / test with no page in two. RVL-CDIP and RVL-CDIP-N images are out, as is any
model trained on them. The corpus is not chosen yet, so the repo holds no data, OCR cache, models
or results. The plan is `~/Downloads/layout-classify_plan.md` (version 1, 2026-10-17). It is not a
claude.ai artifact. It runs in six steps, one Claude Code prompt each. Step 1 has data and the
OCR cache. Steps 2–5 add the classifiers (TF-IDF + LR, UltraBERT-Embed-150M + LR, dit-base + LR,
layoutlmv3-base, gemma4:e4b on page images). Step 6 is the README.

The code is all Python. The repository layout, file header and `test.sh` rules in `~/CLAUDE.md` do
not apply here. Its Python / ML environment section does. Python files open with a module docstring
saying why the file exists, as in synth-classify.

## Commands

```sh
uv sync
uv run python fetch.py                          # source PDFs to data/pdf/ and its manifest; resumable
uv run python quality.py                        # quality gate on each PDF's labelled page
uv run pytest                                   # the one test: classes.py vs aharley/rvl_cdip labels
uv run pytest test_classes.py::test_classes_cover_rvl_cdip_labels
uv run python ocr.py                            # Tesseract on every manifest page; resumable
uv run python train.py --tfidf --train real     # or --embed, --dit; or --train small
uv run python finetune.py --layoutlmv3 --train real   # or --train small; ~11 min real, ~4 small
uv run python score.py --model tfidf-real --split dev --dump   # also test
uv run python confidence.py --model tfidf-real  # needs the dev and test dumps
uv run python score.py --llm gemma4-image --split dev   # or gemma4-image-reject; needs ollama serve
uv run python llm.py --split train              # gemma4:e4b labels to data/llm-labels/; ~4 min
uv run python train.py --tfidf --train real --labels llm   # or --embed; name gets -llm
```

Model names are `tfidf-real`, `tfidf-small`, `ultrabert-embed-real`, `ultrabert-embed-small`,
`dit-real`, `dit-small`, `layoutlmv3-real`, `layoutlmv3-small`, and with `--labels llm`
`tfidf-real-llm` and `ultrabert-embed-real-llm`; `score.py --llm` takes
`gemma4-image` and `gemma4-image-reject`, gemma4:e4b on Ollama given the page image (`llm.py`), with
no score and so no `--dump`. `dit-` models read `data/<split>/<id>.png`, not `ocr/`.

Tesseract comes from Homebrew (`brew install tesseract`). Anything that reaches Hugging Face fails
inside the Claude Code sandbox with `OSStatus -26276`. That is macOS TLS trust lookup through
`truststore`. Run those commands, including the test, outside the sandbox.

There is no linter and no `test.sh`. As in synth-classify, the check on a change is the end-to-end
run.

## How the pieces connect

- **`classes.py` is the single source of the label space.** `CLASSES` holds the 9 types and their
  one-sentence definitions. The definitions are the only type description a prompt-based classifier
  gets. `REJECT_CLASSES` holds the other 7 RVL-CDIP types, of which RVL-CDIP-N has handwritten,
  specification and resume. `REJECT_LABEL = "none of these"`. Names are RVL-CDIP's own label
  strings, with spaces (`scientific report`), not identifiers.
- **No data builder yet.** The scripts read `data/manifest.jsonl`, one row per page with `id`,
  `split` and `label`; `id` starts with its split and names both `data/<split>/<id>.png` and
  `ocr/<id>.json`. Splits are disjoint. The scripts still call the middle split `dev`.
  `test_classes.py` still checks `classes.py` against `aharley/rvl_cdip`'s 16 names. Git ignores the
  page images (`data/*/`); the manifest and selection are tracked.
- **`ocr.py` is the only OCR path.** It reads the manifest, writes `ocr/<id>.json` (`text` plus
  word boxes in page pixels) and skips files already present. Every text classifier reads this
  cache and never re-OCRs. `results/ocr-time.txt` is the one OCR timing, reported apart from
  classifier seconds.
- **`train.py` draws `small` itself.** `--train small` takes 10 train pages per class, one seeded
  `random.Random` per type, and writes the ids to `data/selection.json` under `small`. A rerun puts
  back the same entry.
- **One scoring script, one confidence script.** `score.py` scores every page of a split, reject
  pages included; models predict only `CLASSES`, so a `REJECT_LABEL` page is always wrong.
  `confidence.py` chooses cut-offs on all dev pages and reports, on test, in-class coverage and
  error, `reject_leak`, and a stream with `REJECT_SHARE` (10%) reject pages.
  A later classifier (image or LLM) adds a branch to `score.py`'s `predict` and writes the same dump
  format: `id`, `label`, `predicted`, `score`.
- **`results/`** follows synth-classify's naming without its class-set level:
  `results/<split>/<name>.txt`, `results/train/<name>.txt`, `results/confidence/<name>.txt` and
  `results/confidence/dumps/<name>-<split>.jsonl`. Every number reported needs a file there.

## Constraints from the plan

- Test is read once, after every cut-off and choice has been fixed on dev.
- Exclude `microsoft/dit-base-finetuned-rvlcdip`: it was trained on RVL-CDIP test pages.
- Each step's prompt says: no branches, no commits, no push. Leave work uncommitted for Peter.
- Indra OCR (`prompts/2026-10-17-indra-ocr-pass.md`, a hand-off from the indra session) adds a
  second OCR path. The step 1 brief forbids that, so the hand-off is open pending Peter's decision.
