"""jev_grid.py writes results/jev/score-grid.txt, the file behind the README's account of why Jev's
2% coverage stops at the pages scoring 1.00. Jev's scores come rounded to 0.01, so its cut-offs can
only fall on that grid, and confidence.py's report gives the chosen `tau_cp` but not the bound at
the steps it passed over. This lists, for every 0.01 step from 1.00 down to the 5% `tau_cp`, the
pool pages accepted, how many of them are wrong, and the one-sided 90% Clopper-Pearson bound from
confidence.py's `cp_upper`; then the test reject pages that pass the 2% `tau_cp`, with their ids
and predicted types. It reads the two dumps score.py wrote and makes no call."""

import json
import os

import numpy as np

from classes import CLASSES, REJECT_LABEL
from confidence import CP_LEVEL, cp_upper, read_dump, threshold

NAME = "jev-definitions-s150-reject"
POOL = f"results/confidence/dumps/{NAME}-pool.jsonl"
TEST = f"results/confidence/dumps/{NAME}-test.jsonl"
OUT = "results/jev/score-grid.txt"


def main():
    names = list(CLASSES) + [REJECT_LABEL]
    score, correct, _ = read_dump(POOL, names)
    tau_2 = threshold(score, correct, 0.02, bound=True)
    tau_5 = threshold(score, correct, 0.05, bound=True)

    lines = [
        f"model     {NAME}",
        f"pool      {POOL} ({len(score)} pages)",
        f"test      {TEST}",
        f"pool tau_cp at 2% {tau_2:.2f}, at 5% {tau_5:.2f}; CP level {CP_LEVEL:.0%}, one-sided",
        "pool, accepted at score >= step; a page predicted none is never accepted",
        "  step  accepted  errors  error   cp_upper",
    ]
    # Integer hundredths, so each step is the same float as a dump's two-decimal score.
    for k in range(100, int(round(tau_5 * 100)) - 1, -1):
        step = k / 100
        kept = score >= step
        n = int(kept.sum())
        errors = int((1 - correct[kept]).sum())
        error = f"{errors / n:.4f}" if n else "-"
        bound = f"{cp_upper(errors, n):.4f}" if n else "-"
        lines.append(f"  {step:.2f}  {n:8d}  {errors:6d}  {error:6}  {bound}")

    with open(TEST) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    leaks = [
        r for r in rows
        if r["label"] == REJECT_LABEL and r["predicted"] != REJECT_LABEL and r["score"] >= tau_2
    ]
    lines.append(f"test reject pages given a type with score >= the 2% tau_cp {tau_2:.2f}: {len(leaks)}")
    for r in leaks:
        lines.append(f"  {r['id']}  predicted {r['predicted']}  score {r['score']:.2f}")

    text = "\n".join(lines) + "\n"
    print(text, end="")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
