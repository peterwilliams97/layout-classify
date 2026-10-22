# layout-classify

[synth-classify](../synth-classify/README.md) measured local text classifiers on short messages.
This repo asks the same questions of document pages: native digital documents, or good scans of
them, rendered here at 200 dpi, plus pages that fit none of the types. It holds two classifiers
trained on the OCR text, a local decision model asked about the OCR text with no training, one
classifier on the page image, one fine-tuned on both, and a local vision LLM given the page image
and no hand labels. It holds the corpus manifest, the page split, the OCR cache and the results of
every one of those classifiers.

## Aim

Four questions, measured on one laptop:

1. **How much work can it do without a person checking?** The classifier gives each page a type and
   a score. Above a cut-off, the type is used and nobody reads the page.
   *Coverage* is the share of test pages above the cut-off.
   *Error* is the share of those whose type is wrong.
   We report coverage at 5% error, with 10 hand-labelled pages per class and with every in-class
   train and validation page.
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

**Handwritten removed.** RVL-CDIP-N's twelfth type, handwritten, is not a class here: handwriting
is a medium, not a document type, and its pages are outside the corpus's target of native digital
documents and good scans. `split.py` drops its 131 kept PDFs before splitting, so 11 types remain.
The results with handwritten as a twelfth class, on a split of 906 pages, are at `5fef286`.

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
with its largest image at 150 dpi or more. Every type is gated, handwritten included: 1045 pass and
184 fail, 129 scans under 150 dpi, 53 pages with neither a text layer nor an image at 150 dpi, and 2
that pdfplumber cannot read (`data/pdf/quality.jsonl`).

| Type                   | Kept | Dropped |
|------------------------|------|---------|
| budget                 | 50   | 8       |
| email                  | 30   | 2       |
| form                   | 64   | 4       |
| handwritten (dropped)  | 131  | 40      |
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

**Splits.** `split.py` splits the 685 kept N PDFs of the 11 types by class with `random.Random(0)`:
test 40%, validation 30%, train 30%, each class rounded by largest remainder, a tie going to test,
then validation, then train. A PDF gives one page, so no document is in two splits, and the run
asserts that no sha256 is. It writes `data/manifest.jsonl` (`id`, `split`, `label`, `pdf`, `page`,
`sha256`, and from `render.py` `width` and `height`) and `data/fewshot.txt`, the first 3 train pages
of each class, 33 in all, a subset of train and not a fourth split. `render.py` writes each page to
`data/<split>/<id>.png`.

| Class                  | Train | Validation | Test | Total |
|------------------------|-------|------------|------|-------|
| budget                 | 15    | 15         | 20   | 50    |
| email                  | 9     | 9          | 12   | 30    |
| form                   | 19    | 19         | 26   | 64    |
| invoice                | 13    | 14         | 18   | 45    |
| letter                 | 37    | 37         | 49   | 123   |
| memo                   | 12    | 13         | 17   | 42    |
| news_article           | 18    | 19         | 25   | 62    |
| questionnaire          | 12    | 12         | 15   | 39    |
| resume                 | 39    | 39         | 53   | 131   |
| scientific_publication | 12    | 12         | 15   | 39    |
| specification          | 18    | 18         | 24   | 60    |
| **In-class**           | 204   | 207        | 274  | 685   |
| reject (RVL-CDIP-O)    | 69    | 69         | 91   | 229   |
| **Split total**        | 273   | 276        | 365  | 914   |

The classes share one `random.Random(0)`, drawn in sorted order, so dropping handwritten, which
sorted between form and invoice, changed every later class's draw: budget, email and form kept
their splits, and 333 of the other 685 in-class pages moved split against `5fef286`. The per-class
counts are unchanged.

**Reject pages.** Every kept O PDF, all 229, gives its page 1 as a reject page, labelled `none`,
split as one more type: test 40%, validation 30%, train 30% by largest remainder, drawn with the
same `random.Random(0)` after the classes. The in-class pages are therefore those of `eb3f793`, and
`split.py` exits rather than write a manifest whose in-class rows differ from that commit's. Reject
pages are a quarter of each split. Train's 69 are in the manifest but no classifier trains on them.
The O PDFs come out of the shuffle in the same order as at `eb3f793`, which took the first 30 for
test and the next 46 for validation and train; all 76 are test pages now.

**Limits.**

- Phone photos of documents pass the gate: a photo at 150 dpi or more is a scan by its rule.
- email, questionnaire and scientific_publication have 12 to 16 test pages each (12, 15 and 15), so
  their per-class figures are coarse.
- DocumentCloud documents keep their own rights; CC BY 3.0 covers the dataset, not them.
- DiT and LayoutLMv3 are pretrained on IIT-CDIP, which does not contain these documents.

## Results

On the OCR text, the page image, or both. A page is accepted when its score is at least the cut-off
`tau_cp`, chosen so that the one-sided 90% Clopper-Pearson upper bound on the accepted error is at
most 5%, or 2% (`confidence.py`). Coverage is the share of all 365 test pages accepted, the 91
reject pages included, so it is at most 274 / 365, 0.7507; error is the share of accepted pages
whose type is wrong, a reject page always wrong; reject leak is the share of the 91 reject pages
accepted. The results with 76 reject pages, 30 of them in test, are at `eb3f793`.

**How the cut-off is set.** One rule for every row: the cut-off is fitted on the 549 non-test pages,
the 411 in-class train and validation pages and the 138 reject pages beside them
(`confidence.py --calibration pool`), and test is read once, after. A classifier with no training,
d3-flash, scores the 549 pages as they are (`score.py --split pool`); `fewshot` leaves out its 33
example pages, so its cut-off is fitted on 516. A trained classifier's 549 scores are out of fold
(`train.py --folds 5 --fold-seed 0`, `finetune.py --folds 5 --fold-seed 0`): the 411 in-class pages
are split 5 ways within each class and the 138 reject pages 5 ways apart from them, and each fold is
scored by a model trained on the other 4 folds' in-class pages, 328 or 329 of them, or for small 10
per type drawn from those. No reject page is trained on, and every classifier stays 11-class. C for
UltraBERT-Embed-150M and DiT, and LayoutLMv3's epoch, are chosen on the 549 out-of-fold scores
(`results/train/<name>-pool.txt`). There is no final model: the 5 fold models whose scores set the
cut-off are the ones scored on test (`score.py --split test --folds`), each applies the cut-off to
its own test scores, and a trained row's test figure is the mean over the 5, with the lowest and
highest in brackets. At `70f648e` test was scored instead by one model trained on all 411 pages,
which scored higher than the fold models had: LayoutLMv3 reached 0.9781 accuracy and 0.0337 error
at 2%, 8 of 30 reject pages passing a cut-off no pool reject page had passed. The same figures with
cut-offs fitted on the 230 validation pages alone, and models trained on train alone, are in
[docs/history/2026-10-21-validation-230.md](docs/history/2026-10-21-validation-230.md).

### Hardware.
The results in this file were measured on one
MacBook Pro (Mac17,8): Apple M5 Pro, 18 CPU cores, 48 GB unified memory, no discrete GPU.
UltraBERT, DiT, LayoutLMv3 and d3-flash run on the M5 Pro’s GPU through PyTorch MPS in bfloat16;
gemma4:e4b runs through Ollama; TF-IDF and Tesseract run on the CPU. Nothing else was running.

**Coverage certified under 5% and under 2% error, all test pages**

| Classifier                                  | Labels                     | Coverage at 5%         | Error at 5%            | Reject leak at 5%      | Coverage at 2%         | Error at 2%            | Reject leak at 2%      |
|---------------------------------------------|----------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|
| TF-IDF + LR, real                           | 411 (328–329 per model)    | 0.4975 (0.4904–0.5041) | 0.0386 (0.0276–0.0447) | 0.0264 (0.0220–0.0330) | 0.3342 (0.3205–0.3507) | 0.0213 (0.0163–0.0256) | 0.0198 (0.0110–0.0220) |
| TF-IDF + LR, small                          | 254 of 411 (110 per model) | 0.3518 (0.3370–0.3781) | 0.0311 (0.0236–0.0407) | 0.0308 (0.0220–0.0330) | 0.0000 (0.0000–0.0000) | —                      | 0.0000 (0.0000–0.0000) |
| UltraBERT-Embed-150M + LR, real             | 411 (328–329 per model)    | 0.4718 (0.4548–0.4849) | 0.0174 (0.0113–0.0229) | 0.0264 (0.0220–0.0330) | 0.0000 (0.0000–0.0000) | —                      | 0.0000 (0.0000–0.0000) |
| UltraBERT-Embed-150M + LR, small            | 254 of 411 (110 per model) | 0.3047 (0.2685–0.3589) | 0.0358 (0.0204–0.0619) | 0.0242 (0.0110–0.0330) | 0.0000 (0.0000–0.0000) | —                      | 0.0000 (0.0000–0.0000) |
| d3-flash, `definitions`, 150 words          | 0                          | 0.6438                 | 0.0298                 | 0.0769 (7 of 91)       | 0.5397                 | 0.0102                 | 0.0220 (2 of 91)       |
| d3-flash, `definitions` + `none`, 150 words | 0                          | 0.7205                 | 0.0380                 | 0.1099 (10 of 91)      | 0.6110                 | 0.0000                 | 0.0000 (0 of 91)       |
| d3-flash, `fewshot`, 150 / 50 words         | 33                         | 0.5753                 | 0.0429                 | 0.0879 (8 of 91)       | 0.4575                 | 0.0000                 | 0.0000 (0 of 91)       |
| DiT + LR, real, image                       | 411 (328–329 per model)    | 0.0000 (0.0000–0.0000) | —                      | 0.0000 (0.0000–0.0000) | 0.0000 (0.0000–0.0000) | —                      | 0.0000 (0.0000–0.0000) |
| DiT + LR, small, image                      | 254 of 411 (110 per model) | 0.0000 (0.0000–0.0000) | —                      | 0.0000 (0.0000–0.0000) | 0.0000 (0.0000–0.0000) | —                      | 0.0000 (0.0000–0.0000) |
| LayoutLMv3, real, image and OCR words       | 411 (328–329 per model)    | 0.3742 (0.2932–0.4219) | 0.0341 (0.0187–0.0519) | 0.0396 (0.0000–0.0769) | 0.0000 (0.0000–0.0000) | —                      | 0.0000 (0.0000–0.0000) |
| LayoutLMv3, small, image and OCR words      | 254 of 411 (110 per model) | 0.0000 (0.0000–0.0000) | —                      | 0.0000 (0.0000–0.0000) | 0.0000 (0.0000–0.0000) | —                      | 0.0000 (0.0000–0.0000) |
| gemma4:e4b, image                           | 0                          | —                      | —                      | —                      | —                      | —                      | —                      |
| gemma4:e4b, image + `none`                  | 0                          | —                      | —                      | —                      | —                      | —                      | —                      |

From `results/confidence/<name>-pool.txt`, `tau_cp` lines, "all pages"; a trained row is the mean
over its 5 fold models (min–max), from the summary at the head of that file. Real fold models are
trained on 328 or 329 of the 411 in-class pool pages; small ones on 10 per type drawn from those,
110 pages, 254 distinct pages over the 5. An error's mean is over the fold models that accept a
page; at these cut-offs every trained row accepts pages on all 5 fold models or on none. The bound
holds on the pool, not on test. On the mean, only TF-IDF + LR real at 2%, 0.0213, lands above its
level on all test pages; its in-class error is 0.0067, and the excess is the 1 or 2 of 91 reject
pages each fold model accepts. Single fold models land above too: UltraBERT-Embed-150M + LR small
at 5% up to 0.0619, LayoutLMv3 real at 5% up to 0.0519, and TF-IDF + LR real at 2% up to 0.0256.
UltraBERT-Embed-150M + LR real and small, TF-IDF + LR small and LayoutLMv3 real certify nothing at
2%. LayoutLMv3 real's fold models were retrained here, and on MPS the run is not bit-for-bit
repeatable: its pooled out-of-fold accuracy peaks at epoch 10 here, against epoch 9 at `eb3f793`
on the same in-class training pages. DiT and LayoutLMv3 small certify nothing at either level.
gemma4:e4b's rows are "—" because it answers with a type name and no probability (`llm.py`,
Ollama's structured output), so there is no score to set a cut-off on and `confidence.py` has
nothing to route.

**Accuracy and seconds per page, test**

| Classifier                       | Accuracy, 274 in-class test pages | Seconds per page, median | Seconds per page, mean |
|----------------------------------|-----------------------------------|--------------------------|------------------------|
| TF-IDF + LR, real                | 0.8693 (0.8613–0.8759)            | 0.000329                 | 0.000363               |
| TF-IDF + LR, small               | 0.8869 (0.8686–0.8978)            | 0.00028                  | 0.000302               |
| UltraBERT-Embed-150M + LR, real  | 0.9358 (0.9270–0.9453)            | 0.0355                   | 0.0336                 |
| UltraBERT-Embed-150M + LR, small | 0.8869 (0.8650–0.9088)            | 0.0353                   | 0.0334                 |
| d3-flash, `definitions`          | 0.9818                            | 1.68                     | 1.72                   |
| d3-flash, `definitions` + `none` | 0.9708                            | 1.72                     | 1.82                   |
| d3-flash, `fewshot`              | 0.9197                            | 11.5                     | 11.5                   |
| DiT + LR, real                   | 0.6117 (0.5876–0.6350)            | 0.0189                   | 0.0192                 |
| DiT + LR, small                  | 0.4781 (0.4416–0.4964)            | 0.0192                   | 0.0195                 |
| LayoutLMv3, real                 | 0.9686 (0.9635–0.9745)            | 0.0507                   | 0.0445                 |
| LayoutLMv3, small                | 0.3839 (0.3394–0.4161)            | 0.049                    | 0.0435                 |
| gemma4:e4b, image                | 0.9781                            | 2.65                     | 2.63                   |
| gemma4:e4b, image + `none`       | 0.9708                            | 2.51                     | 2.51                   |

The trained rows are from `results/test/<name>-folds.txt`: the mean accuracy over the 5 fold models
(min–max), each fold's report in `results/test/<name>-fold<f>.txt`, and the seconds median and mean
over all 5 × 365 page scorings. Names are `tfidf-real`, `tfidf-small`, `ultrabert-embed-real`,
`ultrabert-embed-small`, `dit-real`, `dit-small`, `layoutlmv3-real` and `layoutlmv3-small`. The
other rows are from `results/test/<name>.txt`, names `gemma4-image`, `gemma4-image-reject`,
`d3-flash-definitions-s150`, `d3-flash-definitions-s150-reject` and `d3-flash-fewshot-s150-e50`.
d3-flash's answers are cached by page id (`results/d3/cache.jsonl`), so its seconds are those of
the run that first asked about each page. Seconds are the classifier alone, one
page per call, on MPS for UltraBERT, DiT, LayoutLMv3 and d3-flash and through Ollama for
gemma4:e4b; OCR and the image read are not in them. Tesseract takes 0.673 s a page median and 0.795
mean (`results/ocr-time.txt`), to be added for every classifier that reads the OCR text, LayoutLMv3
included; DiT and gemma4:e4b read only the image. Fine-tuning LayoutLMv3 took 10.1 to 11.2 minutes
a fold on real and 4.2 to 10.1 on small, every epoch's checkpoint saved; the epoch kept was 10 of
10 for real and 9 for small (`results/train/layoutlmv3-real-pool.txt`,
`results/train/layoutlmv3-small-pool.txt`). DiT embedded the 549 pool pages in 19.06 s, base model
load included, and every fold and C after that took under half a second
(`results/train/dit-real-pool.txt`).

gemma4:e4b ran on Ollama 0.40.8, to which the Ollama app updated itself on 2026-10-22, before this
run. At `eb3f793` it ran on an earlier Ollama, its version not recorded, with the same model
(`dc35e8d9c606`), and on the same 274 in-class pages it scored 0.9489, at 0.985 s median. Its
change in accuracy and seconds is the runtime's, not the split's.

d3-flash's 5 in-class test errors with `definitions` are a budget and a news article taken for
memos and three specifications, one taken for a letter and two for news articles
(`results/test/d3-flash-definitions-s150.txt`). With `fewshot`, 14 of its 22 errors are
scientific publications, 6 put under news_article and 8 under specification, so only 1 of 15 is
right (`results/test/d3-flash-fewshot-s150-e50.txt`).

**How much the cut-off moves with validation size**

| Classifier                         | Coverage at 5%, 141 (3 draws) | 276    | 549    | Coverage at 2%, 141 (3 draws) | 276    | 549    | Reject leak at 5%, 141 (3 draws) | 276    | 549    |
|------------------------------------|-------------------------------|--------|--------|-------------------------------|--------|--------|----------------------------------|--------|--------|
| d3-flash, `definitions`, 150 words | 0.6712–0.8164                 | 0.6274 | 0.6438 | 0.0000–0.0000                 | 0.5589 | 0.5397 | 0.1319–0.4615                    | 0.0769 | 0.0769 |
| d3-flash, `definitions` + `none`   | 0.7014–0.7616                 | 0.7014 | 0.7205 | 0.0000–0.7397                 | 0.5973 | 0.6110 | 0.0989–0.1758                    | 0.0989 | 0.1099 |
| UltraBERT-Embed-150M + LR, real    | 0.0000–0.6027                 | 0.2685 | 0.4718 | 0.0000–0.0000                 | 0.0000 | 0.0000 | 0.0000–0.0989                    | 0.0000 | 0.0264 |

All on the 365 test pages, the `tau_cp` lines, "all pages". 141: `confidence.py --dev-size 144
--dev-seed S`, S = 0, 1, 2, 12 validation pages per type and 12 reject pages, 141 because email has
9, from `results/confidence/<name>-dev144-seed<S>.txt`. 276: every validation page,
`results/confidence/<name>.txt`. 549: the pool, `results/confidence/<name>-pool.txt`. Names are
`d3-flash-definitions-s150`, `d3-flash-definitions-s150-reject` and `ultrabert-embed-real`. The
141 and 276 columns of the UltraBERT row are the model trained on the 204 train pages
(`results/train/ultrabert-embed-real.txt`, `results/test/ultrabert-embed-real.txt`); its 549
column is the mean over the 5 fold models, so that column moves with the model as well as the
cut-off.

At 141 pages the cut-off swings with the draw: d3-flash `definitions` covers 0.6712 to 0.8164 at
5%, and the draw that covers most lets 42 of 91 reject pages through. At 276 and 549 it settles,
0.6274 and 0.6438, with 7 of 91 through. At 2% 141 pages certify nothing for `definitions` on any
draw, 276 certify 0.5589 and 549 certify 0.5397, with 2 of 91 reject pages through and an error of
0.0102 on test. With `none` as a 12th option, 549 pages certify 0.6110 at 2% with no reject page
through (`results/confidence/d3-flash-definitions-s150-reject-pool.txt`); one 141-page draw
certifies 0.7397 at 2% and on test lets 12 of 91 reject pages through, for an error of 0.0444.
UltraBERT-Embed-150M + LR certifies nothing at 2% at any size; at 5% it covers 0.2685 on 276
validation pages and 0.4718 on the pool.

**1. How much work can it do without a person checking?** With no labelled page, d3-flash
`definitions` + `none` passes the most, 0.7205 of all test pages at 5% and 0.6110 at 2%, of at most
0.7507. With every in-class train and validation page, TF-IDF + LR real passes 0.4975 and 0.3342,
though at 2% its test error is 0.0213, and UltraBERT-Embed-150M + LR real 0.4718 and nothing. With
10 labelled pages per type, TF-IDF + LR small passes 0.3518 at 5% and nothing at 2%, and
UltraBERT-Embed-150M + LR small 0.3047 and nothing. The image rows pass little: DiT and LayoutLMv3
small nothing, LayoutLMv3 real 0.3742 at 5% and nothing at 2%, and gemma4:e4b nothing, having no
score.

**2. Does the page image beat OCR text?** The image alone does not; the image with the OCR words
does on accuracy, on 328 or 329 training pages, but not on coverage. DiT + LR on the image alone
reaches 0.6117 accuracy and certifies no coverage. LayoutLMv3, which reads the image and the OCR
words and boxes together, reaches 0.9686, above UltraBERT-Embed-150M + LR's 0.9358 on the text and
under d3-flash `definitions`' 0.9818, and certifies 0.3742 at 5%, from 0.2932 to 0.4219 by fold
model, against 0.4718 for UltraBERT-Embed-150M + LR and 0.6438 for d3-flash, and nothing at 2%. Its
5% cut-off, 0.961473, is above every fold model's mean score on its right in-class test
predictions, 0.9413 to 0.9523, so small differences between the fold models move many pages across
it. gemma4:e4b on the image is 0.9781 accurate and needs no OCR, but it has no score, so none of its
pages can be passed without a person; 5 of its 6 errors on in-class pages are specifications
(`results/test/gemma4-image.txt`). On seconds, gemma4:e4b's 2.65 s median is above Tesseract's
0.673 s plus UltraBERT's 0.0355 s, and above d3-flash `definitions`' 1.68 s plus OCR. Trained on
small, both image models fall further behind the text ones: 0.4781 and 0.3839 accuracy, no
coverage, against 0.8869 and 0.3047 for UltraBERT small.

**3. What do pages that fit no type cost?** With `none` allowed, gemma4:e4b labels 39 of the 91
reject pages `none` (0.4286) and 4 of the 274 in-class pages (0.0146), all 4 specifications
(`results/test/gemma4-image-reject.txt`). The other 52 reject pages get a type, as all 91 do without
`none`. Over all 365 test pages, where a `none` on a reject page is right, its accuracy is 0.8356
against 0.7342 without `none`. d3-flash `definitions` with `none` labels 65 of the 91 reject pages
`none` (0.7143) and 8 of the 274 in-class pages (0.0292), 6 of them specifications
(`results/test/d3-flash-definitions-s150-reject.txt`). Its accepted in-class pages have no error
with or without `none`, so every accepted error is a reject page. Without `none`, 7 of 91 reject
pages pass at 5% and 2 at 2%, taking in-class coverage of 0.8321 and 0.7117 to 0.6438 and 0.5397 of
all pages, at error 0.0298 and 0.0102 (`results/confidence/d3-flash-definitions-s150-pool.txt`).
With `none`, 10 and 0 pass, taking 0.9234 and 0.8139 to 0.7205 and 0.6110, at error 0.0380 and
0.0000 (`results/confidence/d3-flash-definitions-s150-reject-pool.txt`): `none` raises coverage,
lets 3 more reject pages through at 5% and stops the last 2 at 2%. Reject pages in the pool also
cost coverage of the pages that fit a type, because a cut-off that keeps 138 pool reject pages out
sits higher than one that keeps 46 out. On the same 274 in-class test pages and the same in-class
training, in-class coverage at 5% is 0.6540 for TF-IDF + LR real against 0.7431 at `eb3f793`,
0.6197 against 0.8343 for UltraBERT-Embed-150M + LR real, and 0.8321 against 0.9015 for d3-flash
`definitions`; UltraBERT-Embed-150M + LR real's 2% cut-off, which covered 0.7073 of in-class pages
there, now covers none. Reject leak falls with it: at 5%, 0.0264 of 91 for TF-IDF + LR real and
UltraBERT-Embed-150M + LR real against 0.0600 and 0.0733 of 30, and 0.0769 against 0.3000 for
d3-flash `definitions`.

**4. How many seconds per page?** On the classifier alone, the median is under 0.0004 s for TF-IDF +
LR, 0.0189 for DiT + LR, 0.0355 for UltraBERT-Embed-150M + LR and 0.0507 for LayoutLMv3, against
2.65 for gemma4:e4b on the image and, for d3-flash, 1.68 with `definitions`, 1.72 with `none` and
11.5 with `fewshot`. Every classifier that reads the OCR text adds Tesseract's 0.673 s median
(`results/ocr-time.txt`), so on medians a page through UltraBERT-Embed-150M + LR takes 0.709 s,
through d3-flash `definitions` + `none` 2.39 s, and through gemma4:e4b, with no OCR, 2.65 s.

## Files

- `classes.py` — `CLASSES`, 11 of the 12 RVL-CDIP-N types, handwritten left out, with a one-sentence
  definition each, saying what separates the type from its near pair; `REJECT_LABEL`, `"none"`.
  `test_classes.py` checks that `CLASSES` are the labels of the N rows of `data/pdf/manifest.jsonl`
  less handwritten.
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
  `small`; email has 9 train pages and gives all 9, so small is 109 pages. Neither holds a reject
  page: train's 69 are left out. Every classifier uses `class_weight="balanced"`. For `--embed` and
  `--dit`, C is the value in `C_GRID` (0.01, 0.1, 1, 10), each fitted on train, with the highest
  validation AUROC of top probability against correctness over all validation pages, a tie going to
  the smaller C; `results/train/<name>.txt` gives each value's AUROC and the choice. `--tfidf` keeps
  its fixed C of 10 on its sparse features, with no grid. `--folds 5 --fold-seed 0` trains on the
  pool instead, every train and validation page (Results, How the cut-off is set): out-of-fold
  scores to `results/confidence/dumps/<name>-pool.jsonl`, C chosen on their AUROC, each fold's model
  at that C in `models/<name>/fold<f>/`, small's draws in `data/selection.json` under `small-pool`,
  and `results/train/<name>-pool.txt`. No model is fitted on the whole pool. `--dit` reads the page
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
  are `layoutlmv3-real` and `layoutlmv3-small`. `--folds 5 --fold-seed 0` is `train.py`'s, the
  epoch chosen on pooled out-of-fold accuracy: every fold saves a checkpoint after each epoch, and
  once the epoch is chosen its checkpoint becomes `models/<name>/fold<f>/` and the rest are
  deleted.
- `score.py` — scores a model on every page of validation or test, reject pages included. A model
  predicts one of `CLASSES`, so a reject page is always wrong. A `dit-` model reads
  `data/<split>/<id>.png`; its seconds leave out the image read. A `layoutlmv3-` model reads the OCR
  words and boxes and the image; its seconds are the forward pass, the page read and the processor
  excluded. `--dump` writes each page's prediction and top probability for `confidence.py`.
  `--split pool`, for `--d3` only, scores every train and validation page. `--model <name> --split
  test --folds` scores each fold model of `--folds` in `models/<name>/fold<f>/`, writing
  `results/test/<name>-fold<f>.txt` and `results/confidence/dumps/<name>-fold<f>-test.jsonl`, and
  `results/test/<name>-folds.txt`: the mean accuracy over the folds with its range, and seconds
  over every fold's pages.
- `llm.py` — `classify` sends one page image, `data/<split>/<id>.png`, to `gemma4:e4b` on Ollama
  with `PROMPT`: the 11 `CLASSES` names and their definitions. The reply is restricted by Ollama's
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
  probability for every type. Copied from synth-classify's. The question's options are the 11
  `CLASSES` and its criteria their definitions; with `--prompt fewshot` each definition is followed
  by the OCR text of that type's 3 `data/fewshot.txt` pages, each cut to its first 300 words. The
  state is the page's OCR text cut to its first 1,500 words. `--state-words N` and, with
  `fewshot`, `--example-words N` change those cuts and add `-s<N>` and `-e<N>` to the name.
  `--reject-option` adds `none` as a 12th option, with the criterion "A page that fits none of the
  11 types.", and `-reject` to the name; a `none` answer is right on a reject page and wrong on an
  in-class one, as for `gemma4-image-reject`, and the report adds the same two shares. Every answer
  is cached in
  `results/d3/cache.jsonl`, so a rerun of `score.py --d3` rebuilds its report without the model.
  `score.py --d3 vllm-sr/d3-flash` scores it; with `fewshot`, a `data/fewshot.txt` page in the split
  would be left out, and as those are train pages none is. Its score is the chosen type's
  probability, and its seconds are the `system_one` call. It needs transformers 5.17.0, so it runs
  in `.venv-d3`, built from `requirements-d3.txt`, with `HF_HOME` at `data/hf`, a symlink to
  synth-classify's `data/hf`, which holds the model's 16 GB. Names are `d3-flash-definitions` and
  `d3-flash-fewshot`, with `-s<N>`, `-e<N>` and `-reject` added as above. `--classes six` asks with
  the 6 types with the most test pages as the options — resume, letter, form, news_article,
  specification and budget, `CLASS_SETS` in `d3.py` — scores only pages labelled one of them
  or `none`, and adds `-c6`; `--pages six` scores the same pages with all 11 options and adds
  `-on6`. The option set is part of the cache key.
- `confidence.py` — adapted from synth-classify's. For α 0.02, 0.05 and 0.10 it picks two cut-offs
  on every validation page: `tau`, the lowest score whose accepted validation error is at most α,
  and `tau_cp`, the same with that error's one-sided 90% Clopper-Pearson upper bound. A reject page
  given a type and scoring at or above a cut-off is an accepted error. A page predicted `none` is
  never accepted, whatever its score: it goes to a person, and AUROC, calibration and mean scores
  are over the other pages. It then reports on test, at each cut-off,
  coverage and error on in-class pages, the reject leak (the share of reject pages at or above it),
  and coverage and error for a stream with 10% reject pages and for all test pages. `--dev-size N
  --dev-seed S` fits on N validation pages, N / 12 from each class and from the reject pages, and
  writes `results/confidence/<name>-dev<N>-seed<S>.txt`. `--calibration pool` fits on
  `<name>-pool.jsonl`, the 549 train and validation pages, and writes
  `results/confidence/<name>-pool.txt`; when `score.py --folds` has written fold dumps, the pool
  cut-offs are applied to each, and the file opens with the mean over the fold models of coverage,
  error and reject leak at each cut-off, with min–max, then each fold's full report.
  `test_confidence.py` checks it on hand-made dumps.
- `data/` — the manifest and the selection are tracked. The page images are ignored by git.
- `ocr/` — the OCR cache, one JSON file per page.
- `models/` — trained classifiers, ignored by git and rebuilt by `train.py` and `finetune.py`.
- `results/` — every reported number, in synth-classify's layout without its class-set level:
  `results/<split>/<name>.txt` for score reports (`results/pool/` for `--split pool`,
  `results/test/<name>-fold<f>.txt` and `<name>-folds.txt` for `--folds`),
  `results/train/<name>.txt` and `results/train/<name>-pool.txt` for training time,
  `results/confidence/<name>.txt` and `<name>-pool.txt` for routing, with their inputs in
  `results/confidence/dumps/`, and `results/ocr-time.txt` for OCR time.

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

uv run python train.py --tfidf --train real --folds 5 --fold-seed 0          # also --train small; --embed, --dit
uv run python finetune.py --layoutlmv3 --train real --folds 5 --fold-seed 0  # also --train small
uv run python score.py --model tfidf-real --split test --folds   # the 5 fold models; dumps always written
uv run python confidence.py --model tfidf-real --calibration pool
ollama serve &   # gemma4:e4b pulled
uv run python score.py --llm gemma4-image --split test   # also gemma4-image-reject; validation
uv run python llm.py --split train                       # gemma4:e4b labels train
uv run python train.py --tfidf --train real --labels llm # also --embed

# d3-flash: its own venv, and synth-classify's Hugging Face cache, where the model already is
ln -s ../../synth-classify/data/hf data/hf
uv venv --python 3.13 .venv-d3
uv pip install --python .venv-d3/bin/python -r requirements-d3.txt
export HF_HOME=$PWD/data/hf HF_HUB_OFFLINE=1
for split in pool test; do      # also --prompt fewshot --example-words 50; --reject-option; validation
  .venv-d3/bin/python score.py --d3 vllm-sr/d3-flash --prompt definitions --state-words 150 --split $split --dump
done
.venv-d3/bin/python confidence.py --model d3-flash-definitions-s150 --calibration pool
```

`--embed`, `--dit` and `finetune.py` training and scoring fetch the model from Hugging Face on
first use.

## Licences

layout-classify is MIT; see `LICENSE`. `microsoft/dit-base` is MIT, per
its model card. `microsoft/layoutlmv3-base` is under Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0), per its model card, so the
`layoutlmv3-` models are for non-commercial use only. `gemma4:e4b` is under the Apache License 2.0,
per the licence Ollama ships with it (`ollama show gemma4:e4b --license`).
