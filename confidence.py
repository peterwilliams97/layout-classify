"""Selective-routing report for one classifier.

Input: the two `score.py --dump` files of one model, validation and test, at
results/confidence/dumps/<model>-<split>.jsonl. Each row has `id`, `label`, `predicted` and
`score`, the top probability over CLASSES. A row labelled REJECT_LABEL is wrong unless it was
predicted REJECT_LABEL, so a reject page given a type that scores above a cut-off is an accepted
error. A model that may predict REJECT_LABEL (score.py's `gemma4-image-reject` and
`--reject-option`) is right on a reject page it so labels and wrong on an in-class one, and a page
predicted REJECT_LABEL is never accepted, whatever its score: it goes to a person. Its score is
read as -inf, below every cut-off, and AUROC, isotonic regression, ECE and mean scores are over
the other pages.

Computed:
  AUROC of score against correctness on every validation page.
  For each alpha in ALPHAS, two cut-offs from validation, every validation page included:
    tau     the lowest validation score at which the validation error among pages
            scoring >= tau is <= alpha;
    tau_cp  the same, with the validation error replaced by its one-sided
            CP_LEVEL Clopper-Pearson upper bound.
  Then on test, the same cut-offs applied:
    on in-class pages (label in CLASSES): AUROC of score against
    correctness; isotonic regression of correctness on score, fitted on
    validation, and ECE over ECE_BINS equal-count bins, before and after;
    coverage (share of pages scoring >= cut-off) and error among them;
    at tau, recall per true class among accepted pages;
    reject_leak, the share of REJECT_LABEL pages scoring >= cut-off, also
    given on validation;
    coverage and error of a stream that is REJECT_SHARE reject pages and
    the rest in-class, from the in-class numbers and reject_leak:
      coverage_10 = (1 - r) * coverage_inclass + r * reject_leak
      error_10    = (r * reject_leak
                     + (1 - r) * coverage_inclass * error_inclass) / coverage_10
    with r = REJECT_SHARE; and coverage and error over all test pages, at
    test's own mix of reject pages.

Output: the report, printed and written to results/confidence/<model>.txt, or with --dev-size to
results/confidence/<model>-dev<N>-seed<S>.txt.

With --calibration pool, the cut-offs and the calibration are fitted on <model>-pool.jsonl instead
of the validation dump: every train and validation page, from score.py --split pool for a model
with no training, or the out-of-fold scores of train.py's or finetune.py's --folds for a trained
one. Test is read as without it. The report goes to results/confidence/<model>-pool.txt, and the
word "validation" in its first part reads "pool". When score.py --folds has written fold dumps,
<model>-fold<f>-test.jsonl, the pool cut-offs are applied to each of them instead of one
<model>-test.jsonl: the report opens with, for each alpha and cut-off, the mean over the fold models
of coverage, error and reject_leak, in-class and over all test pages, with their range, and then
gives each fold's report in full.

With --dev-size N, the cut-offs and the calibration are fitted on N validation pages drawn by
`draw`, N / 12 from each class of CLASSES and from the reject pages, so the cut-off's dependence on
validation size can be measured; a class with fewer pages gives them all. --dev-seed seeds the
draw.

Nothing is fitted or chosen on test. Adapted from synth-classify's confidence.py.
"""

import argparse
import glob
import json
import os
import random
import sys

import numpy as np
from scipy.stats import beta
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from classes import CLASSES, REJECT_LABEL

ALPHAS = [0.02, 0.05, 0.10]
ECE_BINS = 10
# One-sided confidence level of the Clopper-Pearson upper bound on validation error behind `tau_cp`.
CP_LEVEL = 0.90
# Share of reject pages in the stream the deployed numbers are reported for. Test's own share is
# set by the corpus's counts, not by any expected input.
REJECT_SHARE = 0.10
DUMPS = "results/confidence/dumps"


def read_dump(path, names):
    """Reads a --dump file. Returns three arrays over its rows: score, -inf where the prediction is
    REJECT_LABEL; correct (predicted == label, as 0/1); and label index in `names`. Exits if a
    label or a prediction is not in `names`."""
    with open(path) as f: rows = [json.loads(line) for line in f if line.strip()]
    if not rows: sys.exit(f"{path}: no rows")
    for r in rows:
        if r["label"] not in names or r["predicted"] not in names:
            sys.exit(f"{path}: row not over CLASSES and REJECT_LABEL: {r['id']}")
    score = np.array([-np.inf if r["predicted"] == REJECT_LABEL else r["score"] for r in rows])
    correct = np.array([r["predicted"] == r["label"] for r in rows], dtype=float)
    label = np.array([names.index(r["label"]) for r in rows])
    return score, correct, label


def draw(label, size, seed, classes):
    """draw returns the indices of `size` // `classes` rows of each class in `label`, each class's
    rows shuffled in class order by one random.Random(`seed`); a class with fewer rows gives all of
    them."""
    rng = random.Random(seed)
    picked = []
    for i in range(classes):
        rows = [j for j in range(len(label)) if label[j] == i]
        rng.shuffle(rows)
        picked += rows[: size // classes]
    return np.array(sorted(picked), dtype=int)


def ece(conf, correct, raw):
    """Expected calibration error of `conf` against `correct` over ECE_BINS equal-count bins. Rows
    are ordered by `conf`, ties broken by `raw`."""
    order = np.lexsort((raw, conf))
    total = 0.0
    for b in np.array_split(order, ECE_BINS):
        total += len(b) / len(order) * abs(conf[b].mean() - correct[b].mean())
    return total


def cp_upper(errors, n):
    """One-sided CP_LEVEL Clopper-Pearson upper bound on an error rate of `errors` in `n`; 1.0 when
    errors == n."""
    return 1.0 if errors == n else float(beta.ppf(CP_LEVEL, errors + 1, n - errors))


def threshold(score, correct, alpha, bound):
    """Lowest value in `score` at which the error of `correct` among rows with score >= that value
    is <= `alpha`. With `bound`, the error is replaced by cp_upper of it. Returns inf if no value
    qualifies. A -inf score is never a cut-off, so its row is never accepted."""
    for tau in np.unique(score[np.isfinite(score)]):
        kept = score >= tau
        errors, n = int((1 - correct[kept]).sum()), int(kept.sum())
        if (cp_upper(errors, n) if bound else errors / n) <= alpha:
            return tau
    return np.inf


def routed(score, correct, tau):
    """Coverage of `tau` (share of rows with score >= tau) and the error among those rows; the error
    is None when no row qualifies."""
    kept = score >= tau
    return kept.mean(), (1 - correct[kept]).mean() if kept.any() else None


def at_cut_off(score, correct, label, tau):
    """at_cut_off returns the test figures of cut-off `tau` over dump arrays from `read_dump`:
    in-class coverage and error, reject_leak, the REJECT_SHARE stream's coverage and error, and
    coverage and error over every row. An error is None when no row is accepted."""
    reject = label == len(CLASSES)
    cov, err = routed(score[~reject], correct[~reject], tau)
    leak = (score[reject] >= tau).mean()
    cov_10 = (1 - REJECT_SHARE) * cov + REJECT_SHARE * leak
    err_10 = None
    if cov_10 > 0:
        err_10 = ((1 - REJECT_SHARE) * cov * (err or 0) + REJECT_SHARE * leak) / cov_10
    all_cov, all_err = routed(score, correct, tau)
    return cov, err, leak, cov_10, err_10, all_cov, all_err


def cut_offs(score, correct):
    """cut_offs returns `tau` and `tau_cp` for each alpha in ALPHAS, fitted on `score` and
    `correct`, keyed as the report prints them."""
    return {
        alpha: {
            "tau   ": threshold(score, correct, alpha, bound=False),
            "tau_cp": threshold(score, correct, alpha, bound=True),
        }
        for alpha in ALPHAS
    }


def folds_report(model, pool_path, test_paths):
    """folds_report returns the --calibration pool report for a model scored on test by score.py
    --folds: the pool cut-offs applied to each dump of `test_paths`, one per fold model, summarised
    as the mean and range over the folds, then `report` for each fold. An error's mean is over the
    folds that accept a page, and the line says how many do."""
    names = list(CLASSES) + [REJECT_LABEL]
    pool_s, pool_c, _ = read_dump(pool_path, names)
    cuts = cut_offs(pool_s, pool_c)
    folds = [read_dump(path, names) for path in test_paths]
    n = len(folds)

    def spread(xs, of=n):
        xs = [x for x in xs if x is not None]
        if not xs:
            return f"{'-':<6} (no fold accepts a page)"
        text = f"{np.mean(xs):.4f} ({min(xs):.4f}–{max(xs):.4f})"
        return text if len(xs) == of else text + f", {len(xs)} of {of} folds accept a page"

    lines = [
        f"model     {model}, {n} fold models",
        f"pool      {pool_path}",
        *(f"test      {path}" for path in test_paths),
        f"cut-offs from the pool pages, applied to each fold model's test dump; mean over the {n}"
        " fold models (min–max)",
    ]
    for alpha, taus in cuts.items():
        lines.append(f"  alpha {alpha:.2f}")
        for label, t in taus.items():
            figures = [at_cut_off(s, c, y, t) for s, c, y in folds]
            n_rej = [int((y == len(CLASSES)).sum()) for _, _, y in folds]
            leak_n = [int((s[y == len(CLASSES)] >= t).sum()) for s, _, y in folds]
            lines += [
                f"    {label} {fmt(None if np.isinf(t) else t, '.6f')}",
                f"      in-class    coverage {spread([f[0] for f in figures])}",
                f"      in-class    error    {spread([f[1] for f in figures])}",
                f"      reject_leak          {spread([f[2] for f in figures])},"
                f" {'/'.join(map(str, leak_n))} of {n_rej[0]} by fold",
                f"      all pages   coverage {spread([f[5] for f in figures])}",
                f"      all pages   error    {spread([f[6] for f in figures])}",
            ]
    sections = [
        report(f"{model}, {os.path.basename(path)}", pool_path, path, fit="pool")
        for path in test_paths
    ]
    return "\n".join(lines) + "\n\n" + "\n".join(sections)


def fmt(x, spec=".4f"):
    return "-" if x is None else format(x, spec)


def report(model, validation_path, test_path, dev_size=None, dev_seed=0, fit="validation"):
    """report returns the text of the routing report for `model` from its validation and test
    dumps. With `dev_size`, the cut-offs and the calibration are fitted on `draw`'s rows of the
    validation dump. `fit` names the pages of `validation_path` in the report: "validation", or
    "pool" for --calibration pool."""
    names = list(CLASSES) + [REJECT_LABEL]
    reject = names.index(REJECT_LABEL)
    width = max(len(name) for name in names)

    validation_s, validation_c, validation_y = read_dump(validation_path, names)
    validation_desc = f"{len(validation_s)} pages"
    if dev_size is not None:
        keep = draw(validation_y, dev_size, dev_seed, len(names))
        validation_desc = f"{len(keep)} of {len(validation_s)} pages, drawn with seed {dev_seed}"
        validation_s, validation_c, validation_y = (
            validation_s[keep], validation_c[keep], validation_y[keep]
        )

    scored = np.isfinite(validation_s)
    iso = IsotonicRegression(increasing=True, y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(validation_s[scored], validation_c[scored])
    cuts = cut_offs(validation_s, validation_c)
    validation_reject = validation_y == reject
    all_s, all_c, all_y = read_dump(test_path, names)
    held = not (scored.all() and np.isfinite(all_s).all())

    lines = [
        f"model     {model}",
        f"{fit} {validation_path} ({validation_desc},"
        f" {int(validation_reject.sum())} reject; accuracy {validation_c.mean():.4f})",
        "score     top probability over CLASSES; a REJECT_LABEL page is always wrong"
        if not held else
        f"score     top probability; a page predicted {REJECT_LABEL}, right only on a reject page,"
        " is never accepted",
        f"{fit} AUROC of score against correctness, all pages"
        + (f" not predicted {REJECT_LABEL}" if held else "")
        + f"  {roc_auc_score(validation_c[scored], validation_s[scored]):.4f}",
        f"routing: accept a page when its score is at least the cut-off, chosen on the {fit}"
        " pages above",
        f"  tau     lowest {fit} score whose accepted {fit} error is at most alpha",
        f"  tau_cp  the same with the {fit} error's one-sided {CP_LEVEL:.0%} Clopper-Pearson"
        " upper bound",
        "  reject_leak  share of reject pages scoring at least the cut-off",
    ]
    for alpha, taus in cuts.items():
        lines.append(f"  alpha {alpha:.2f}")
        for label, t in taus.items():
            cov, err = routed(validation_s, validation_c, t)
            leak = (validation_s[validation_reject] >= t).mean()
            lines.append(
                f"    {label} {fmt(None if np.isinf(t) else t, '.6f'):<8}  "
                f"{fit} coverage {cov:.4f} error {fmt(err)} reject_leak {leak:.4f}"
            )

    test_reject = all_y == reject
    if not test_reject.any():
        sys.exit(f"{test_path}: no reject pages, so reject_leak is undefined")
    s, c, y = all_s[~test_reject], all_c[~test_reject], all_y[~test_reject]
    # `fs` and `fc` are the in-class pages with a score, for the lines that need one.
    fs, fc = s[np.isfinite(s)], c[np.isfinite(s)]
    if not len(fc):
        sys.exit(f"{test_path}, in-class: every page is predicted {REJECT_LABEL}")
    # With every scored in-class page right, as when a model's only in-class errors are
    # REJECT_LABEL, AUROC and the mean wrong score are undefined and print as "-".
    auroc = roc_auc_score(fc, fs) if 0 < fc.sum() < len(fc) else None
    wrong_mean = fs[fc == 0].mean() if (fc == 0).any() else None
    right_mean = fs[fc == 1].mean() if fc.any() else None
    n_rej = int(test_reject.sum())
    lines += [
        "",
        f"test: {test_path} ({len(all_s)} pages, {n_rej} reject)",
        f"  in-class pages {len(s)}, accuracy {c.mean():.4f}",
    ]
    if held:
        lines += [
            f"  predicted {REJECT_LABEL}, never accepted: {int(np.isinf(s).sum())} of {len(s)}"
            f" in-class pages, {int(np.isinf(all_s[test_reject]).sum())} of {n_rej} reject pages;"
            " the four lines below are over the rest of the in-class pages",
        ]
    lines += [
        f"  AUROC of score against correctness, in-class  {fmt(auroc)}",
        f"  mean score, right predictions       {fmt(right_mean)} ({int(fc.sum())})",
        f"  mean score, wrong predictions       {fmt(wrong_mean)} ({int(len(fc) - fc.sum())})",
        f"  ECE, {ECE_BINS} equal-count bins, isotonic fitted on {fit}"
        f"  before {ece(fs, fc, fs):.4f}  after {ece(iso.predict(fs), fc, fs):.4f}",
        f"  stream_10: {REJECT_SHARE:.0%} reject pages, {1 - REJECT_SHARE:.0%} in-class",
    ]
    for alpha, taus in cuts.items():
        lines.append(f"  alpha {alpha:.2f}")
        for label, t in taus.items():
            cov, err, leak, cov_10, err_10, all_cov, all_err = at_cut_off(all_s, all_c, all_y, t)
            lines += [
                f"    {label} {fmt(None if np.isinf(t) else t, '.6f')}",
                f"      in-class             coverage {cov:.4f} error {fmt(err)}",
                f"      reject_leak          {leak:.4f} ({int((all_s[test_reject] >= t).sum())}"
                f" of {n_rej})",
                f"      stream_10            coverage {cov_10:.4f} error {fmt(err_10)}",
                f"      all pages, {n_rej} of {len(all_s)} reject"
                f"  coverage {all_cov:.4f} error {fmt(all_err)}",
            ]
        lines.append("    recall among accepted at tau, in-class, per true class"
                     " (accepted of total)")
        kept = s >= taus["tau   "]
        for i, name in enumerate(CLASSES):
            mine = y == i
            if not mine.any():
                continue
            n = int((kept & mine).sum())
            recall = c[kept & mine].mean() if n else None
            lines.append(f"      {name:<{width}} {fmt(recall)} ({n} of {int(mine.sum())})")

    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model name, as given to score.py")
    ap.add_argument("--dev-size", type=int, metavar="N", help="fit on N validation pages, stratified")
    ap.add_argument("--dev-seed", type=int, default=0, metavar="S", help="seed of the --dev-size draw")
    ap.add_argument("--calibration", choices=["validation", "pool"], default="validation",
                    help="pages the cut-offs are fitted on; pool is train and validation")
    args = ap.parse_args()
    if args.calibration == "pool" and args.dev_size is not None:
        ap.error("--dev-size draws from validation; it does not go with --calibration pool")
    classes = len(CLASSES) + 1
    if args.dev_size is not None and (args.dev_size <= 0 or args.dev_size % classes):
        ap.error(f"--dev-size must be a positive multiple of {classes}, CLASSES and REJECT_LABEL")
    path = lambda split: os.path.join(DUMPS, f"{args.model}-{split}.jsonl")
    folds = sorted(
        glob.glob(path("fold*-test")),
        key=lambda p: int(p.rsplit("-fold", 1)[1].split("-")[0]),
    )
    if args.calibration == "pool" and folds:
        text = folds_report(args.model, path("pool"), folds)
    else:
        text = report(
            args.model,
            path(args.calibration),
            path("test"),
            args.dev_size,
            args.dev_seed,
            args.calibration,
        )
    sys.stdout.write(text)
    name = args.model
    if args.calibration == "pool":
        name += "-pool"
    if args.dev_size is not None:
        name += f"-dev{args.dev_size}-seed{args.dev_seed}"
    out = os.path.join("results", "confidence", name + ".txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(text)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
