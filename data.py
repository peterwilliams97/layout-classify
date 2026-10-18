"""data.py builds the page sets every classifier here reads: train, dev and test, all drawn with
SEED from RVL-CDIP-N in SOURCE_DIR, written as data/<split>/<id>.png with one row per page in
data/manifest.jsonl and the drawn file names in data/selection.json.

RVL-CDIP-N is modern pages from the web and DocumentCloud, labelled with RVL-CDIP's types. Its file
names are `<source>_<type>.tif`, the type's spaces written as underscores; the type is read from
there. Each type's files are shuffled with a generator of their own, seeded from SEED and the type.

A type in CLASSES is split by FRACTIONS: train takes the floor of half, dev the floor of a fifth,
and test the remainder. A type in REJECT_CLASSES gives no train pages, DEV_REJECTS dev pages, and a
share of TEST_REJECTS test pages, split as evenly as the types RVL-CDIP-N has allow, the first in
REJECT_CLASSES order taking one more. Its remaining pages are listed under "unused".

An `rvl_index` is the RVL-CDIP-N file name. A page that fails to decode is logged, listed in
data/selection.json, and its place taken by the next page in the same seeded order."""

import json
import os
import random
import sys

from PIL import Image

from classes import CLASSES, REJECT_CLASSES, REJECT_LABEL

SOURCE_DIR = os.path.expanduser("~/data/rvl-cdip-n/RVL-CDIP-N")
SEED = 0
SPLITS = ["train", "dev", "test"]
# Share of an in-class type's pages that goes to train and to dev; test takes the rest.
FRACTIONS = {"train": 0.5, "dev": 0.2}
DEV_REJECTS = 20
TEST_REJECTS = 100


def source_label(filename):
    """The RVL-CDIP type named by an RVL-CDIP-N file name's suffix, or None if the suffix names no
    type."""
    stem = filename.rsplit(".", 1)[0]
    found = [n for n in list(CLASSES) + REJECT_CLASSES if stem.endswith("_" + n.replace(" ", "_"))]
    return found[0] if len(found) == 1 else None


def test_reject_counts(present):
    """Splits TEST_REJECTS over the reject types in `present`, in REJECT_CLASSES order, the
    remainder going to the first."""
    base, extra = divmod(TEST_REJECTS, len(present))
    return {name: base + (i < extra) for i, name in enumerate(present)}


def decode(path):
    """The image at `path` with its pixels loaded, or the error that stopped it."""
    try:
        img = Image.open(path)
        img.load()
        return img, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def main():
    files = sorted(f for f in os.listdir(SOURCE_DIR) if f.lower().endswith((".tif", ".tiff")))
    if not files:
        sys.exit(f"{SOURCE_DIR}: no TIFF files")
    by_type = {}
    for f in files:
        rvl_label = source_label(f)
        if rvl_label is None:
            sys.exit(f"{SOURCE_DIR}/{f}: file name names no RVL-CDIP type")
        by_type.setdefault(rvl_label, []).append(f)
    missing = [name for name in CLASSES if name not in by_type]
    if missing:
        sys.exit(f"{SOURCE_DIR}: no pages of {missing}")
    present_rejects = [name for name in REJECT_CLASSES if name in by_type]
    test_rejects = test_reject_counts(present_rejects)

    for split in SPLITS:
        os.makedirs(f"data/{split}", exist_ok=True)
    manifest, failures, short = [], {}, []
    selection = {split: {} for split in SPLITS}
    unused = {}
    for rvl_label in list(CLASSES) + present_rejects:
        order = list(by_type[rvl_label])
        random.Random(f"{SEED}/{rvl_label}").shuffle(order)
        in_class = rvl_label in CLASSES
        need = len(order) if in_class else DEV_REJECTS + test_rejects[rvl_label]

        # Decode in seeded order until `need` pages have decoded; a failure's place goes to the
        # next file.
        kept, images, n = [], {}, 0
        while len(kept) < need and n < len(order):
            f = order[n]
            n += 1
            img, err = decode(os.path.join(SOURCE_DIR, f))
            if err:
                failures[f] = err
                print(f"decode failure: {f}: {err}", file=sys.stderr)
                continue
            kept.append(f)
            images[f] = img
        if len(kept) < need:
            short.append(f"{rvl_label}: {len(kept)} of {need}")

        if in_class:
            n_train = int(len(kept) * FRACTIONS["train"])
            n_dev = int(len(kept) * FRACTIONS["dev"])
            parts = {"train": kept[:n_train], "dev": kept[n_train : n_train + n_dev],
                     "test": kept[n_train + n_dev :]}
            label = rvl_label
        else:
            parts = {"train": [], "dev": kept[:DEV_REJECTS], "test": kept[DEV_REJECTS:]}
            unused[rvl_label] = order[n:]
            label = REJECT_LABEL

        for split in SPLITS:
            selection[split][rvl_label] = parts[split]
            for f in parts[split]:
                page_id = f"{split}-{f.rsplit('.', 1)[0]}"
                images[f].save(f"data/{split}/{page_id}.png", format="PNG")
                manifest.append({"id": page_id, "split": split, "label": label,
                                 "rvl_label": rvl_label, "rvl_index": f})

    with open("data/manifest.jsonl", "w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")
    with open("data/selection.json", "w") as f:
        json.dump({"seed": SEED, "source": SOURCE_DIR.replace(os.path.expanduser("~"), "~"),
                   "splits": selection, "unused": unused,
                   "decode_failures": dict(sorted(failures.items()))}, f, indent=1)
        f.write("\n")

    width = max(len(name) for name in by_type)
    print(f"{'':<{width}} " + " ".join(f"{s:>5}" for s in SPLITS) + " unused", file=sys.stderr)
    for rvl_label in list(CLASSES) + present_rejects:
        counts = " ".join(f"{len(selection[s][rvl_label]):>5}" for s in SPLITS)
        print(f"{rvl_label:<{width}} {counts} {len(unused.get(rvl_label, [])):>6}", file=sys.stderr)
    for split in SPLITS:
        rows = [r for r in manifest if r["split"] == split]
        rej = sum(r["label"] == REJECT_LABEL for r in rows)
        print(f"{split}: {len(rows)} pages, {rej} reject", file=sys.stderr)
    print(f"decode failures: {len(failures)}", file=sys.stderr)
    if short:
        sys.exit("short of pages: " + "; ".join(short))


if __name__ == "__main__":
    main()
