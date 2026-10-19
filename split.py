"""split.py draws the train / validation / test split from the source PDFs and writes
data/manifest.jsonl, one row per page, and data/fewshot.txt. It is the only place a page is given a
split, so no other script can put one document in two.

In-class pages are the labelled page of every RVL-CDIP-N PDF in data/pdf/manifest.jsonl that is not
an alias and that quality.py kept. Each class is split on its own: its PDFs, in file-name order, are
shuffled, and the first go to test, the next to validation and the rest to train, in the shares of
SHARES. A share's count is rounded by largest remainder, a tie going to test, then validation, then
train. Reject pages are page 1 of kept RVL-CDIP-O PDFs, labelled REJECT_LABEL: round(in-class / 9)
per split, so they are a tenth of the split, drawn from the shuffled O PDFs in the order test,
validation, train. One random.Random(0) makes every shuffle, classes in sorted order, then O.

A manifest row is `id`, `split`, `label`, `pdf` (the file under data/pdf/), `page` (0-based) and
`sha256`; `id` is the split and the file stem, so it names data/<split>/<id>.png and ocr/<id>.json.
data/fewshot.txt lists the first FEWSHOT train ids of each class in manifest order, one per line: a
subset of train, not a fourth split. The run asserts that no sha256 is in two splits and prints the
per-class table."""

import collections
import json
import math
import random

from classes import CLASSES, REJECT_LABEL

SOURCE = "data/pdf/manifest.jsonl"
QUALITY = "data/pdf/quality.jsonl"
MANIFEST = "data/manifest.jsonl"
FEWSHOT_FILE = "data/fewshot.txt"
SPLITS = ["train", "validation", "test"]
SHARES = {"test": 0.4, "validation": 0.3, "train": 0.3}  # dict order breaks remainder ties
FEWSHOT = 3


def quotas(n):
    """The number of `n` PDFs each split gets, by largest remainder over SHARES."""
    exact = {s: n * share for s, share in SHARES.items()}
    counts = {s: math.floor(x) for s, x in exact.items()}
    order = sorted(SHARES, key=lambda s: -(exact[s] - counts[s]))  # stable: ties keep SHARES order
    for s in order[:n - sum(counts.values())]:
        counts[s] += 1
    return counts


def assign(rows, counts, rng):
    """`rows` shuffled by `rng` and cut into `counts`, test first; returns (split, row) pairs."""
    rows = sorted(rows, key=lambda r: r["file"])
    rng.shuffle(rows)
    out, i = [], 0
    for s in SHARES:
        out += [(s, r) for r in rows[i:i + counts[s]]]
        i += counts[s]
    return out


def main():
    with open(SOURCE) as f:
        source = {r["file"]: r for r in map(json.loads, f)}
    with open(QUALITY) as f:
        kept = [q["file"] for q in map(json.loads, f) if q["kept"]]
    pool = [source[k] for k in kept if not source[k]["alias_of"]]

    rng = random.Random(0)
    by_class = collections.defaultdict(list)
    for r in pool:
        if r["set"] == "N":
            by_class[r["label"]].append(r)
    if sorted(by_class) != sorted(CLASSES):
        raise SystemExit(f"{SOURCE}: N labels {sorted(by_class)} are not CLASSES")
    picked = []
    for label in sorted(by_class):
        picked += [(s, r, label) for s, r in assign(by_class[label], quotas(len(by_class[label])),
                                                     rng)]
    in_class = collections.Counter(s for s, _, _ in picked)
    reject_counts = {s: round(in_class[s] / 9) for s in SHARES}
    o_rows = [r for r in pool if r["set"] == "O"]
    if sum(reject_counts.values()) > len(o_rows):
        raise SystemExit(f"{sum(reject_counts.values())} reject pages needed, {len(o_rows)} kept")
    picked += [(s, r, REJECT_LABEL) for s, r in assign(o_rows, reject_counts, rng)]

    rows = [{"id": f"{s}-{r['file'].removesuffix('.pdf')}", "split": s, "label": label,
             "pdf": r["file"], "page": r["page"], "sha256": r["sha256"]}
            for s, r, label in picked]
    rows.sort(key=lambda r: (SPLITS.index(r["split"]), r["id"]))

    split_of = {}
    for r in rows:
        other = split_of.setdefault(r["sha256"], r["split"])
        assert other == r["split"], f"{r['sha256']} in {other} and {r['split']}"

    with open(MANIFEST, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    fewshot = []
    for label in CLASSES:
        fewshot += [r["id"] for r in rows if r["split"] == "train" and r["label"] == label][:FEWSHOT]
    with open(FEWSHOT_FILE, "w") as f:
        f.write("\n".join(fewshot) + "\n")

    table = collections.Counter((r["label"], r["split"]) for r in rows)
    print(f"| {'Class':22} | Train | Validation | Test | Total |")
    print(f"|{'-' * 24}|-------|------------|------|-------|")

    def line(name, counts):
        print(f"| {name:22} | {counts[0]:<5} | {counts[1]:<10} | {counts[2]:<4} | "
              f"{sum(counts):<5} |")

    for label in sorted(CLASSES):
        line(label, [table[label, s] for s in SPLITS])
    line("**In-class**", [in_class[s] for s in SPLITS])
    line("reject (RVL-CDIP-O)", [reject_counts[s] for s in SPLITS])
    line("**Split total**", [in_class[s] + reject_counts[s] for s in SPLITS])
    print(f"\n{len(rows)} pages, {len(fewshot)} few-shot ids; reject share "
          + ", ".join(f"{s} {reject_counts[s] / (in_class[s] + reject_counts[s]):.2%}"
                      for s in SPLITS))


if __name__ == "__main__":
    main()
