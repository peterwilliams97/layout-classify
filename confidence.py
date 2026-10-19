"""Selective-routing report for one classifier.

Input: the two `score.py --dump` files of one model, dev and test, at
results/confidence/dumps/<model>-<split>.jsonl. Each row has `id`, `label`, `predicted` and
`score`, the top probability over CLASSES. A row labelled REJECT_LABEL is always wrong: no model
predicts it, so a reject page that scores above a cut-off is an accepted error.

Computed:
  AUROC of score against correctness on every dev page.
  For each alpha in ALPHAS, two cut-offs from dev, every dev page included:
    tau     the lowest dev score at which the dev error among pages
            scoring >= tau is <= alpha;
    tau_cp  the same, with the dev error replaced by its one-sided
            CP_LEVEL Clopper-Pearson upper bound.
  Then on test, the same cut-offs applied:
    on in-class pages (label in CLASSES): AUROC of score against
    correctness; isotonic regression of correctness on score, fitted on
    dev, and ECE over ECE_BINS equal-count bins, before and after;
    coverage (share of pages scoring >= cut-off) and error among them;
    at tau, recall per true class among accepted pages;
    reject_leak, the share of REJECT_LABEL pages scoring >= cut-off, also
    given on dev;
    coverage and error of a stream that is REJECT_SHARE reject pages and
    the rest in-class, from the in-class numbers and reject_leak:
      coverage_10 = (1 - r) * coverage_inclass + r * reject_leak
      error_10    = (r * reject_leak
                     + (1 - r) * coverage_inclass * error_inclass) / coverage_10
    with r = REJECT_SHARE; and coverage and error over all test pages, at
    test's own mix of reject pages.

Output: the report, printed and written to results/confidence/<model>.txt.

Nothing is fitted or chosen on test. Adapted from synth-classify's confidence.py.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import beta
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from classes import CLASSES, REJECT_LABEL

ALPHAS = [0.02, 0.05, 0.10]
ECE_BINS = 10
# One-sided confidence level of the Clopper-Pearson upper bound on dev error behind `tau_cp`.
CP_LEVEL = 0.90
# Share of reject pages in the stream the deployed numbers are reported for. Test's own share is
# set by the corpus's counts, not by any expected input.
REJECT_SHARE = 0.10
DUMPS = "results/confidence/dumps"


def read_dump(path, names):
    """Reads a --dump file. Returns three arrays over its rows: score, correct (predicted == label,
    as 0/1) and label index in `names`. Exits if a label is not in `names` or a prediction is not
    in CLASSES."""
    with open(path) as f: rows = [json.loads(line) for line in f if line.strip()]
    if not rows: sys.exit(f"{path}: no rows")
    for r in rows:
        if r["label"] not in names or r["predicted"] not in CLASSES:
            sys.exit(f"{path}: row not over CLASSES and REJECT_LABEL: {r['id']}")
    score = np.array([r["score"] for r in rows])
    correct = np.array([r["predicted"] == r["label"] for r in rows], dtype=float)
    label = np.array([names.index(r["label"]) for r in rows])
    return score, correct, label


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
    qualifies."""
    for tau in np.unique(score):
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


def fmt(x, spec=".4f"):
    return "-" if x is None else format(x, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model name, as given to score.py")
    args = ap.parse_args()
    names = list(CLASSES) + [REJECT_LABEL]
    reject = names.index(REJECT_LABEL)
    width = max(len(name) for name in names)
    path = lambda split: os.path.join(DUMPS, f"{args.model}-{split}.jsonl")

    dev_s, dev_c, dev_y = read_dump(path("dev"), names)
    iso = IsotonicRegression(increasing=True, y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(dev_s, dev_c)
    cuts = {
        alpha: {
            "tau   ": threshold(dev_s, dev_c, alpha, bound=False),
            "tau_cp": threshold(dev_s, dev_c, alpha, bound=True),
        }
        for alpha in ALPHAS
    }
    dev_reject = dev_y == reject

    lines = [
        f"model     {args.model}",
        f"dev       {path('dev')} ({len(dev_s)} pages, {int(dev_reject.sum())} reject;"
        f" accuracy {dev_c.mean():.4f})",
        "score     top probability over CLASSES; a REJECT_LABEL page is always wrong",
        f"dev AUROC of score against correctness, all pages  {roc_auc_score(dev_c, dev_s):.4f}",
        "routing: accept a page when its score is at least the cut-off, chosen on every dev page",
        "  tau     lowest dev score whose accepted dev error is at most alpha",
        f"  tau_cp  the same with the dev error's one-sided {CP_LEVEL:.0%} Clopper-Pearson upper"
        " bound",
        "  reject_leak  share of reject pages scoring at least the cut-off",
    ]
    for alpha, taus in cuts.items():
        lines.append(f"  alpha {alpha:.2f}")
        for label, t in taus.items():
            cov, err = routed(dev_s, dev_c, t)
            leak = (dev_s[dev_reject] >= t).mean()
            lines.append(
                f"    {label} {fmt(None if np.isinf(t) else t, '.6f'):<8}  "
                f"dev coverage {cov:.4f} error {fmt(err)} reject_leak {leak:.4f}"
            )

    all_s, all_c, all_y = read_dump(path("test"), names)
    test_reject = all_y == reject
    if not test_reject.any():
        sys.exit(f"{path('test')}: no reject pages, so reject_leak is undefined")
    s, c, y = all_s[~test_reject], all_c[~test_reject], all_y[~test_reject]
    if c.all() or not c.any():
        sys.exit(f"{path('test')}, in-class: every prediction is {'right' if c.all() else 'wrong'}")
    n_rej = int(test_reject.sum())
    lines += [
        "",
        f"test: {path('test')} ({len(all_s)} pages, {n_rej} reject)",
        f"  in-class pages {len(s)}, accuracy {c.mean():.4f}",
        f"  AUROC of score against correctness, in-class  {roc_auc_score(c, s):.4f}",
        f"  mean score, right predictions       {s[c == 1].mean():.4f} ({int(c.sum())})",
        f"  mean score, wrong predictions       {s[c == 0].mean():.4f} ({int(len(c) - c.sum())})",
        f"  ECE, {ECE_BINS} equal-count bins, isotonic fitted on dev"
        f"  before {ece(s, c, s):.4f}  after {ece(iso.predict(s), c, s):.4f}",
        f"  stream_10: {REJECT_SHARE:.0%} reject pages, {1 - REJECT_SHARE:.0%} in-class",
    ]
    for alpha, taus in cuts.items():
        lines.append(f"  alpha {alpha:.2f}")
        for label, t in taus.items():
            cov, err = routed(s, c, t)
            leak = (all_s[test_reject] >= t).mean()
            cov_10 = (1 - REJECT_SHARE) * cov + REJECT_SHARE * leak
            err_10 = None
            if cov_10 > 0:
                err_10 = ((1 - REJECT_SHARE) * cov * (err or 0) + REJECT_SHARE * leak) / cov_10
            all_cov, all_err = routed(all_s, all_c, t)
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

    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    out = os.path.join("results", "confidence", args.model + ".txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(text)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
