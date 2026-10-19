# Corpus plan: RVL-CDIP-N source PDFs

Checked 2026-10-19. The corpus is the PDFs RVL-CDIP-N's page images were made from, rendered here at
200 dpi. Nothing beyond listings and 10 one-kilobyte probes has been downloaded.

## What the source holds

- The dataset link in `github.com/gxlarson/rvl-cdip-ood`'s README, `https://tinyurl.com/4he6my23`,
  redirects to the Google Drive folder `1VDnwRhmguvhKUCZ0_nv54RMGgqfYHGfz`. Its top level is
  `evaluation/`, `output/`, `PDFs/`, `RVL-CDIP-N/`, `RVL-CDIP-O/`, `training/`, `LICENSE.txt`,
  `README.md`. `PDFs/` is `pdfs_1/` to `pdfs_9/`, 2408 PDFs in all (`rvl-cdip-ood/pdfs-listing.jsonl`).
- **RVL-CDIP-N: all 1002 images have their PDF**, 1002 distinct files (`rvl-cdip-ood/n-to-pdf.jsonl`).
  An image `…__pdf<k>_<stem>.tif` matches `PDFs/pdfs_*/<stem>.pdf`. For 998 the file name is unique;
  for the other 4 (`documentcloud_00044_memo`, `documentcloud_00116_handwritten`, each in two
  folders) the folder is the image's leading `documentcloud_<k>` or else its `pdf<k>`.
- **RVL-CDIP-O: 254 of its 2915 images have their PDF** by the same match (`rvl-cdip-ood/o-to-pdf.jsonl`).
  The other 2661 images carry no PDF name: 1679 `ood__<n>.tif`, 742 `ood__documentcloud_<n>.tif`,
  the rest hash-named. The 742 numbers all fall inside `pdfs_1/`'s 1143 unlabelled
  `documentcloud_<n>.pdf` (1 to 1143), which suggests those are their PDFs; the match is unverified.
- **jordyvl/RVL-CDIP-N on Hugging Face has no PDFs and no URLs**: five parquet files of `image`
  (JPEG) and `label`, 1002 rows, one split `test`; its README points back to the Drive folder.
- 10 PDFs drawn with `random.Random(0)` from the N matches, fetched as bytes 0–1023 from
  `drive.usercontent.google.com/download?id=<id>`: 10 of 10 returned HTTP 206 and begin `%PDF-1.`
  (`rvl-cdip-ood/url-probe.tsv`).

## Splits

Stratified per class, seed 0, split by PDF, over the 816 N PDFs that are not aliases and pass
`quality.py`'s gate. Test 40%, validation 30%, train 30%, each class rounded by largest remainder, a
tie going to test, then validation, then train. `split.py` draws it and writes `data/manifest.jsonl`.

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

Reject is round(in-class / 9) per split, so 10% of the split: 10.00%, 9.89%, 9.92%. The 90 are
drawn with seed 0 from the 229 kept O PDFs. Few-shot examples are 3 per class from train, 36
pages, a subset of train and not a fourth split (`data/fewshot.txt`).

## Open before the download

- Which page of a multi-page PDF is the labelled page is not recorded. Page 1 is the default; a
  check against the RVL-CDIP-N image is the alternative.
- The 4 basenames in two folders, and any PDF that appears twice under different names, must be
  checked by content hash before the draw, or one document can land in two splits.
- `classes.py` makes handwritten, specification and resume reject types; this plan has all 12 N
  types in-class and takes reject from O.
- Many DocumentCloud PDFs are scans. Whether each meets "good scan" is not checked.
- The dataset is CC BY 3.0; the documents inside keep their own rights.
