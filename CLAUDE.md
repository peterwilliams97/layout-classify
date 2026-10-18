# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this
repository.

## What this is

A worked example, not a library. It is the sibling of `../synth-classify`, which measured local text
classifiers on Banking77. This repo asks the same questions of modern document pages from
RVL-CDIP-N, plus a reject class. RVL-CDIP itself was the first source and was replaced because its
1990s low-resolution scans are not the target input; its results stay in `results/rvl-cdip/` and the
README's appendix. The plan is `~/Downloads/layout-classify_plan.md` (version 1, 2026-10-17). It is
not a claude.ai artifact. It runs in six steps, one Claude Code prompt each. Step 1 has data and the
OCR cache. Steps 2–5 add the classifiers (TF-IDF + LR, UltraBERT-Embed-150M + LR, dit-base + LR,
layoutlmv3-base, gemma4:e4b on page images). Step 6 is the README.

The code is all Python. The repository layout, file header and `test.sh` rules in `~/CLAUDE.md` do
not apply here. Its Python / ML environment section does. Python files open with a module docstring
saying why the file exists, as in synth-classify.

## Commands

```sh
uv sync
uv run pytest                                   # the one test: classes.py vs aharley/rvl_cdip labels
uv run pytest test_classes.py::test_classes_cover_rvl_cdip_labels
uv run python data.py                           # ~12 s; reads ~/data/rvl-cdip-n/RVL-CDIP-N/
uv run python ocr.py                            # Tesseract on every manifest page; ~5 min; resumable
uv run python train.py --tfidf --train real     # or --embed; or --train small
uv run python score.py --model tfidf-real --split dev --dump   # also test
uv run python confidence.py --model tfidf-real  # needs the dev and test dumps
```

Model names are `tfidf-real`, `tfidf-small`, `ultrabert-embed-real`, `ultrabert-embed-small`. An
`--embed` training run is about 13 s for 288 pages on MPS; scoring is about 0.035 s per page.

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
- **`data.py` reads one source, RVL-CDIP-N**, a flat directory of `<source>_<type>.tif` files.
  Each type is shuffled with its own `random.Random(f"{SEED}/{type}")`. A type in `CLASSES` goes
  half to train, a fifth to dev (both rounded down) and the rest to test. A reject type gives 0
  train, 20 dev and 34 or 33 test, 100 in all; the rest are listed under `unused`. A page that fails
  to decode is replaced by the next in seeded order. Splits are disjoint, unlike synth-classify,
  where dev was part of train. `test_classes.py` still checks `classes.py` against
  `aharley/rvl_cdip`'s 16 names.
- **Outputs.** `data/manifest.jsonl` has one row per page: `id`, `split`, `label`, `rvl_label`,
  `rvl_index`. `rvl_index` is the RVL-CDIP-N file name; `id` is `<split>-<file name without .tif>`
  and names both `data/<split>/<id>.png` and `ocr/<id>.json`. `data/selection.json` holds the seed,
  the file names per split and type, `unused` and `decode_failures`. Git ignores the page images
  (`data/*/`); the manifest and selection are tracked.
- **`ocr.py` is the only OCR path.** It reads the manifest, writes `ocr/<id>.json` (`text` plus
  word boxes in page pixels) and skips files already present. Every text classifier reads this
  cache and never re-OCRs. `results/ocr-time.txt` is the one OCR timing, reported apart from
  classifier seconds.
- **`train.py` draws `small` itself.** `--train small` takes 10 train pages per class, one seeded
  `random.Random` per type, and writes the ids to `data/selection.json` under `small`. A `data.py`
  run rewrites that file without the entry; the next `--train small` puts it back unchanged.
- **One scoring script, one confidence script.** `score.py` scores every page of a split, reject
  pages included; models predict only `CLASSES`, so a `REJECT_LABEL` page is always wrong.
  `confidence.py` chooses cut-offs on all dev pages and reports, on test, in-class coverage and
  error, `reject_leak`, and a stream with `REJECT_SHARE` (10%) reject pages.
  A later classifier (image or LLM) adds a branch to `score.py`'s `predict` and writes the same dump
  format: `id`, `label`, `predicted`, `score`.
- **`results/`** follows synth-classify's naming without its class-set level:
  `results/<split>/<name>.txt`, `results/train/<name>.txt`, `results/confidence/<name>.txt` and
  `results/confidence/dumps/<name>-<split>.jsonl`. Every number reported needs a file there.
  `results/unweighted/` holds train, score and confidence files from the run without class weights,
  and `results/rvl-cdip/` the same layout from the RVL-CDIP runs, plus test-n; no script writes to
  either.

## Constraints from the plan

- Test is read once, after every cut-off and choice has been fixed on dev.
- Exclude `microsoft/dit-base-finetuned-rvlcdip`: it was trained on RVL-CDIP test pages.
- RVL-CDIP label noise is reported as a limit, not cleaned.
- Test has about 180 in-class pages, so differences under 5 points are noise.
- Each step's prompt says: no branches, no commits, no push. Leave work uncommitted for Peter.
- Indra OCR (`prompts/2026-10-17-indra-ocr-pass.md`, a hand-off from the indra session) adds a
  second OCR path. The step 1 brief forbids that, so the hand-off is open pending Peter's decision.
