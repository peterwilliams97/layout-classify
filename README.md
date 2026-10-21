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
| reject (RVL-CDIP-O)    | 23    | 23         | 30   | 76    |
| **Split total**        | 227   | 230        | 304  | 761   |

The classes share one `random.Random(0)`, drawn in sorted order, so dropping handwritten, which
sorted between form and invoice, changed every later class's draw: budget, email and form kept
their splits, and 333 of the other 685 in-class pages moved split against `5fef286`. The per-class
counts are unchanged.

**Reject pages.** Each split gets round(in-class / 9) reject pages, a tenth of the split: page 1 of
kept O PDFs, drawn with the same `random.Random(0)` after the classes, labelled `none`. Train's 23
are in the manifest but no classifier trains on them.

**Limits.**

- Phone photos of documents pass the gate: a photo at 150 dpi or more is a scan by its rule.
- email, questionnaire and scientific_publication have 12 to 16 test pages each (12, 15 and 15), so
  their per-class figures are coarse.
- DocumentCloud documents keep their own rights; CC BY 3.0 covers the dataset, not them.
- DiT and LayoutLMv3 are pretrained on IIT-CDIP, which does not contain these documents.

## Results

On the OCR text, the page image, or both. A page is accepted when its score is at least the cut-off
`tau_cp`, chosen on validation so that the one-sided 90% Clopper-Pearson upper bound on the accepted
validation error is at most 5%, or 2% (`confidence.py`). Coverage is the share of all 304 test pages
accepted, the 30 reject pages included; error is the share of accepted pages whose type is wrong, a
reject page always wrong; reject leak is the share of the 30 reject pages accepted. Small is 10
train pages per type, 9 for email.

**Coverage certified under 5% and under 2% error, all test pages**

| Classifier                                  | Labels | Coverage at 5% | Error at 5% | Reject leak at 5% | Coverage at 2% | Error at 2% | Reject leak at 2% |
|---------------------------------------------|--------|----------------|-------------|-------------------|----------------|-------------|-------------------|
| TF-IDF + LR, train                          | 204    | 0.5461         | 0.0542      | 0.0333 (1 of 30)  | 0.0000         | —           | 0.0000 (0 of 30)  |
| TF-IDF + LR, small                          | 109    | 0.3849         | 0.0171      | 0.0333 (1 of 30)  | 0.0000         | —           | 0.0000 (0 of 30)  |
| UltraBERT-Embed-150M + LR, train            | 204    | 0.7632         | 0.0517      | 0.1667 (5 of 30)  | 0.0000         | —           | 0.0000 (0 of 30)  |
| UltraBERT-Embed-150M + LR, small            | 109    | 0.6151         | 0.0267      | 0.0333 (1 of 30)  | 0.0000         | —           | 0.0000 (0 of 30)  |
| d3-flash, `definitions`, 150 words          | 0      | 0.8487         | 0.0349      | 0.3000 (9 of 30)  | 0.6711         | 0.0098      | 0.0667 (2 of 30)  |
| d3-flash, `fewshot`, 150 / 50 words         | 33     | 0.7664         | 0.0515      | 0.3000 (9 of 30)  | 0.6612         | 0.0149      | 0.0667 (2 of 30)  |
| DiT + LR, train, image                      | 204    | 0.0000         | —           | 0.0000 (0 of 30)  | 0.0000         | —           | 0.0000 (0 of 30)  |
| DiT + LR, small, image                      | 109    | 0.0000         | —           | 0.0000 (0 of 30)  | 0.0000         | —           | 0.0000 (0 of 30)  |
| LayoutLMv3, train, image and OCR words      | 204    | 0.0000         | —           | 0.0000 (0 of 30)  | 0.0000         | —           | 0.0000 (0 of 30)  |
| LayoutLMv3, small, image and OCR words      | 109    | 0.0000         | —           | 0.0000 (0 of 30)  | 0.0000         | —           | 0.0000 (0 of 30)  |
| gemma4:e4b, image                           | 0      | —              | —           | —                 | —              | —           | —                 |
| gemma4:e4b, image + `none`                  | 0      | —              | —           | —                 | —              | —           | —                 |

From `results/confidence/<name>.txt`, `tau_cp` lines, "all pages". Validation is 207 in-class
pages, with 23 reject pages beside them. In synth-classify, at 200 validation rows the certified
coverage at 5% ranged from 0 to 0.95 between draws for UltraBERT-Embed-150M + LR, and at 400 from
0.72 to 0.93 (synth-classify `README.md` @ `6368e3b`), so one draw of 230 pages here is not a
stable figure. The bound holds on validation, not on test: three rows land just above 5% on all
test pages, TF-IDF + LR on train at 0.0542, UltraBERT-Embed-150M + LR on train at 0.0517 and
d3-flash `fewshot` at 0.0515, each with an in-class error under 5% (0.0485, 0.0308 and 0.0134) and
the reject pages it accepts on top. At 2% the bound needs the top 114 validation pages right with no
error before it accepts any page; only d3-flash's ranking gives that. gemma4:e4b's rows are "—"
because it answers with a type name and no probability (`llm.py`, Ollama's structured output), so
there is no score to set a cut-off on and `confidence.py` has nothing to route.

**Accuracy and seconds per page, test**

| Classifier                       | Accuracy, 274 in-class test pages | Seconds per page, median | Seconds per page, mean |
|----------------------------------|-----------------------------------|--------------------------|------------------------|
| TF-IDF + LR, train               | 0.8175                            | 0.000308                 | 0.000336               |
| TF-IDF + LR, small               | 0.8321                            | 0.000288                 | 0.000314               |
| UltraBERT-Embed-150M + LR, train | 0.9343                            | 0.0349                   | 0.0336                 |
| UltraBERT-Embed-150M + LR, small | 0.8759                            | 0.0349                   | 0.0335                 |
| d3-flash, `definitions`          | 0.9818                            | 1.68                     | 1.71                   |
| d3-flash, `fewshot`              | 0.9197                            | 11.4                     | 11.4                   |
| DiT + LR, train                  | 0.5401                            | 0.0184                   | 0.0189                 |
| DiT + LR, small                  | 0.4088                            | 0.0186                   | 0.019                  |
| LayoutLMv3, train                | 0.8102                            | 0.0494                   | 0.0436                 |
| LayoutLMv3, small                | 0.3285                            | 0.0492                   | 0.0435                 |
| gemma4:e4b, image                | 0.9489                            | 0.985                    | 1.01                   |
| gemma4:e4b, image + `none`       | 0.9307                            | 0.97                     | 0.98                   |

From `results/test/<name>.txt`; names are `dit-real`, `dit-small`, `layoutlmv3-real`,
`layoutlmv3-small`, `gemma4-image` and `gemma4-image-reject`, and the d3-flash rows are
`d3-flash-definitions-s150` and `d3-flash-fewshot-s150-e50`. Seconds are the classifier alone, one
page per call, on MPS for UltraBERT, DiT, LayoutLMv3 and d3-flash and through Ollama for gemma4:e4b;
OCR and the image read are not in them. Tesseract takes 0.673 s a page median and 0.795 mean
(`results/ocr-time.txt`), to be added for every classifier that reads the OCR text, LayoutLMv3
included; DiT and gemma4:e4b read only the image. Fine-tuning LayoutLMv3 took 7.9 minutes on train
and 5.3 on small, and the epoch kept was 10 of 10 and 6 of 10; on train its validation accuracy was
still rising at epoch 10, 0.7217 to 0.7391 (`results/train/layoutlmv3-real.txt`,
`results/train/layoutlmv3-small.txt`). DiT + LR trained in 18.96 s and 15.39 s
(`results/train/dit-real.txt`, `results/train/dit-small.txt`).

d3-flash's 5 in-class test errors with `definitions` are a budget and a news article taken for
memos and three specifications, one taken for a letter and two for news articles
(`results/test/d3-flash-definitions-s150.txt`). With `fewshot`, 14 of its 22 errors are
scientific publications, 6 put under news_article and 8 under specification, so only 1 of 15 is
right (`results/test/d3-flash-fewshot-s150-e50.txt`).

**What the question and state lengths do**

| Prompt        | Options | State words | Example words | Accuracy, 274 in-class test pages | Coverage at 5% | Reject leak at 5% | Coverage at 2% | Reject leak at 2% | Seconds per page, median |
|---------------|---------|-------------|---------------|-----------------------------------|----------------|-------------------|----------------|-------------------|--------------------------|
| `definitions` | 11      | 1500        | —             | 0.9854                            | 0.8388         | 0.2667 (8 of 30)  | 0.6776         | 0.0667 (2 of 30)  | 2.24                     |
| `definitions` | 11      | 150         | —             | 0.9818                            | 0.8487         | 0.3000 (9 of 30)  | 0.6711         | 0.0667 (2 of 30)  | 1.68                     |
| `fewshot`     | 11      | 150         | 50            | 0.9197                            | 0.7664         | 0.3000 (9 of 30)  | 0.6612         | 0.0667 (2 of 30)  | 11.4                     |
| `definitions` | 12      | 150         | —             | 0.9708                            | 0.9013         | 0.2667 (8 of 30)  | 0.8684         | 0.2000 (6 of 30)  | 1.72                     |
| `definitions` | 11      | 150         | —             | 0.9746 (of 197)                   | 0.8062         | 0.2000 (6 of 30)  | 0.0000         | 0.0000 (0 of 30)  | 1.68                     |
| `definitions` | 6       | 150         | —             | 0.9797 (of 197)                   | 0.8502         | 0.2333 (7 of 30)  | 0.7048         | 0.0333 (1 of 30)  | 1.3                      |

d3-flash, `score.py --d3 --state-words N --example-words N`: the page's OCR text is cut to its
first N state words, and each `fewshot` example to its first N example words. Options are the 11
types, or with `--reject-option` 12, `none` the last; a page labelled `none` is never accepted.
Accuracy and seconds from `results/test/<name>.txt`; coverage and reject leak from the `tau_cp`
lines, "all pages", of `results/confidence/<name>.txt`. Names are `d3-flash-definitions`,
`d3-flash-definitions-s150`, `d3-flash-fewshot-s150-e50`, `d3-flash-definitions-s150-reject`,
`d3-flash-definitions-s150-on6` and `d3-flash-definitions-s150-c6`. The last two rows are on the
6-class subset of test, the 227 pages labelled resume, letter, form, news_article, specification,
budget or `none` (197 in-class, 30 reject), with cut-offs from the 170 validation pages so labelled;
the 11-option row is `d3-flash-definitions-s150`'s answers on those pages, read from the cache, and
the 6-option row asks with those 6 types alone (`score.py --pages six`, `--classes six`).

Two rows of the table with handwritten were not rerun. `fewshot` at a 1,500-word state and 300-word
examples took 51.8 s a page median at `5fef286`, nearly 8 hours for validation and test, and was
dropped; the 150 / 50 row stands for `fewshot`. The 500-word `definitions` row was not in this
rerun either. Both are at `5fef286`, with handwritten as a twelfth class.

Cutting the page costs little: from 1,500 words to 150, in-class accuracy falls from 0.9854 to
0.9818, coverage at 5% moves from 0.8388 to 0.8487 and at 2% from 0.6776 to 0.6711, and the median
page goes from 2.24 s to 1.68 s. The examples still hurt: `fewshot` at 50-word examples is 0.9197
against 0.9818 for `definitions` at the same 150-word state, certifies less at both levels, and
takes 11.4 s a page against 1.68. Every difference in coverage here is a few pages on one
validation draw of 230.

**How much the cut-off moves between validation draws**

| Classifier                         | Coverage at 5%, 3 draws (all 230) | Error at 5%, max | Coverage at 2%, 3 draws (all 230) | Error at 2%, max | Reject leak at 5%, 3 draws |
|------------------------------------|-----------------------------------|------------------|-----------------------------------|------------------|----------------------------|
| d3-flash, `definitions`, 150 words | 0.7928–0.8487 (0.8487)            | 0.0349           | 0.0000–0.0000 (0.6711)            | —                | 0.2000–0.3000              |
| d3-flash, `definitions` + `none`   | 0.8980–0.9013 (0.9013)            | 0.0293           | 0.0000–0.8684 (0.8684)            | 0.0227           | 0.2667–0.2667              |
| UltraBERT-Embed-150M + LR, train   | 0.3651–0.7632 (0.7632)            | 0.0517           | 0.0000–0.0000 (0.0000)            | —                | 0.0000–0.1667              |

`confidence.py --dev-size 144 --dev-seed S`, S = 0, 1, 2: cut-offs fitted on 12 validation pages
per type and 12 reject pages, 141 pages because email has 9, then applied to all 304 test pages;
the `tau_cp` lines, "all pages", of `results/confidence/<name>-dev144-seed<S>.txt`, with the
figure in brackets from `results/confidence/<name>.txt`. Names are `d3-flash-definitions-s150`,
`d3-flash-definitions-s150-reject` and `ultrabert-embed-real`. Error at 2% is "—" where no draw
accepts a page.

At 5%, d3-flash's two rows move little across the draws, 0.7928 to 0.8487 and 0.8980 to 0.9013,
and every draw's error is under 5% on test. UltraBERT-Embed-150M + LR moves from 0.3651 to 0.7632,
so its 0.7632 on all 230 pages is the top of its range, not the middle. At 2% the 141 pages
certify nothing for `definitions` in any draw, against 0.6711 on 230, and for `definitions` with
`none` they certify 0.8684 on two draws and nothing on the third. The ranking at 5% holds in every
draw, `none` first, `definitions` second, UltraBERT third; at 2% it holds on seeds 0 and 2 and
seed 1 ties all three at 0.

**How much depends on the number of choices?** Little of the accuracy, and here fewer choices raise
the coverage. On the 197 in-class test pages of the 6-class subset, 6 options are right on 193 and
11 on 192, 0.9797 against 0.9746 (`results/test/d3-flash-definitions-s150-c6.txt`,
`results/test/d3-flash-definitions-s150-on6.txt`). Fewer options raise the top probability, a mean
of 0.9797 on right answers against 0.9636, and the ranking holds: the 5% cut-off rises from 0.867704
to 0.886821 and coverage still rises from 0.8062 to 0.8502, with 7 reject pages through against 6
of 30. At 2% the 6-option run certifies 0.7048 with 1 of 30 reject pages through, and the 11-option
answers certify nothing: no cut-off on the 170 validation pages has a Clopper-Pearson bound that
low, where all 230 gave 0.6711 with 11 options
(`results/confidence/d3-flash-definitions-s150-c6.txt`,
`results/confidence/d3-flash-definitions-s150-on6.txt`). The median page takes 1.3 s with 6
options against 1.68 s, the shorter question.

**Does a 12th option, `none`, catch reject pages?** Two in three. At the 150-word state d3-flash
labels 20 of the 30 reject test pages `none` (0.6667), against gemma4:e4b's 16 of 30 with `none`
allowed. The cut-off falls with it, to 0.473749 from 0.794289 at 5% and to 0.637405 from 0.964708
at 2%, so of the 10 reject pages given a type, 8 pass at 5% against 9 of 30 with 11 options, and 6
pass at 2% against 2. The cost on in-class pages is 8 of 274 labelled `none` (0.0292), 6 of them
specifications, a budget and a news article, so in-class accuracy falls from 0.9818 to 0.9708;
every in-class page it gives a type is right. Coverage rises, from 0.8487 to 0.9013 at 5% (error
0.0349 to 0.0292) and from 0.6711 to 0.8684 at 2%, but the error at 2% rises from 0.0098 to 0.0227,
above the 2% it was certified for: on validation 1 of 23 reject pages passed that cut-off, on test
6 of 30. The median page takes 1.72 s against 1.68
(`results/test/d3-flash-definitions-s150-reject.txt`,
`results/confidence/d3-flash-definitions-s150-reject.txt`).

With `none` allowed, gemma4:e4b labels 16 of the 30 reject pages `none` (0.5333) and 5 of the 274
in-class pages (0.0182), all 5 specifications (`results/test/gemma4-image-reject.txt`). The other
14 reject pages get a type, as all 30 do without `none`. Over all 304 test pages, where a `none` on
a reject page is right, its accuracy is 0.8914 against 0.8553 without `none`.

**Does the page image beat OCR text?** Not here. Against UltraBERT-Embed-150M + LR on train's OCR
text, 0.7632 coverage at 5% and 0.9343 accuracy, and d3-flash `definitions` on the first 150 words,
0.8487 and 0.9818, every image row with a score is lower on both and certifies no coverage. DiT + LR
on the image alone reaches 0.5401 accuracy; LayoutLMv3, which reads the image and the OCR words and
boxes together, reaches 0.8102. gemma4:e4b on the image is 0.9489 accurate, above UltraBERT and
under d3-flash, and needs no OCR, but it has no score, so none of its pages can be passed without a
person; 6 of its 14 errors on in-class pages are letters and emails labelled memo
(`results/test/gemma4-image.txt`). On seconds, gemma4:e4b's 0.985 s median is above Tesseract's
0.673 s plus UltraBERT's 0.0349 s, and under d3-flash's 1.68 s plus OCR. Trained on small, both
image models fall further behind the text ones: 0.4088 and 0.3285 accuracy, no coverage, against
0.8759 and 0.6151 for UltraBERT small.

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
  page: train's 23 are left out. Every classifier uses `class_weight="balanced"`. For
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
  writes `results/confidence/<name>-dev<N>-seed<S>.txt`. `test_confidence.py` checks it on a
  hand-made 13-row dump.
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
for split in validation test; do      # also --prompt fewshot; --state-words 150 --reject-option; --classes six; --pages six
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
