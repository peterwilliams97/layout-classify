"""data.py builds the page sets every classifier here reads: train, dev and test, drawn with SEED
from RVL-CDIP as published in `aharley/rvl_cdip`, written as data/<split>/<id>.png with one row per
page in data/manifest.jsonl and the drawn RVL-CDIP indices in data/selection.json.

`aharley/rvl_cdip` is a loader script over one 38.8 GB tar.gz holding all three RVL-CDIP splits,
with the split membership and labels in data/{train,val,test}.txt. Current `datasets` refuses
loader scripts, and its streaming mode would read the whole archive once per split. So the draw is
made from the three label lists, and one streamed pass over the archive keeps the drawn pages and
discards everything else; nothing but the selected pages reaches the disk. The pass stops as soon
as every drawn page has gone by.

An `rvl_index` is a line number, from 0, in the label list of the page's RVL-CDIP split. Each draw
takes SPARE pages beyond those it needs, in seeded order, so a page that fails to decode is replaced
by the next one rather than leaving its class short; failures are logged to stderr and listed in
data/selection.json."""

import io
import json
import os
import random
import sys
import tarfile
import time
import urllib.request

from huggingface_hub import hf_hub_download, hf_hub_url
from PIL import Image

from classes import CLASSES, REJECT_CLASSES, REJECT_LABEL

REPO = "aharley/rvl_cdip"
ARCHIVE = "data/rvl-cdip.tar.gz"
# Archive member names are the label lists' paths under this prefix.
MEMBER_PREFIX = "images/"
SEED = 0
# split: (RVL-CDIP split label list, pages per class in CLASSES, reject pages)
SPLITS = {
    "train": ("train", 100, 0),
    "dev":   ("val",   20,  20),
    "test":  ("test",  40,  40),
}
SPARE = 5
PROGRESS_BYTES = 2 << 30


def rvl_label_names():
    """The 16 RVL-CDIP label names in `aharley/rvl_cdip`'s class-id order, read from its
    dataset_infos.json."""
    path = hf_hub_download(REPO, "dataset_infos.json", repo_type="dataset")
    with open(path) as f:
        return json.load(f)["default"]["features"]["label"]["names"]


def label_list(rvl_split):
    """Rows of `aharley/rvl_cdip`'s data/<rvl_split>.txt as (path, class id), in file order."""
    path = hf_hub_download(REPO, f"data/{rvl_split}.txt", repo_type="dataset")
    with open(path) as f:
        rows = [line.split() for line in f if line.strip()]
    return [(p, int(c)) for p, c in rows]


def reject_counts(n):
    """Splits `n` reject pages over REJECT_CLASSES as evenly as possible, the remainder going to the
    first types in list order."""
    base, extra = divmod(n, len(REJECT_CLASSES))
    return {name: base + (i < extra) for i, name in enumerate(REJECT_CLASSES)}


def draw(split, rows, names):
    """Draws one group per class of `split`: (label, rvl_label, pages needed, candidate rvl indices
    in seeded order, SPARE more than needed). Each group has its own seeded generator, so a group's
    draw does not depend on the others."""
    _, per_class, rejects = SPLITS[split]
    wanted = [(name, name, per_class) for name in CLASSES]
    wanted += [(REJECT_LABEL, name, k) for name, k in reject_counts(rejects).items() if k]
    groups = []
    for label, rvl_label, k in wanted:
        pool = [i for i, (_, c) in enumerate(rows) if names[c] == rvl_label]
        rng = random.Random(f"{SEED}/{split}/{rvl_label}")
        groups.append((label, rvl_label, k, rng.sample(pool, k + SPARE)))
    return groups


def stream_pages(paths):
    """One pass over the archive. Returns {path: PNG bytes} for each member in `paths` that decodes
    and {path: error} for each that does not or never appears."""
    url = hf_hub_url(REPO, ARCHIVE, repo_type="dataset")
    pending, pages, failures = set(paths), {}, {}
    start, seen, next_report = time.time(), 0, PROGRESS_BYTES
    with urllib.request.urlopen(url) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for member in tar:
            seen += member.size
            if seen >= next_report:
                print(f"{seen / 2**30:.1f} GiB of members read, {len(pending)} pages to go, "
                      f"{time.time() - start:.0f} s", file=sys.stderr)
                next_report += PROGRESS_BYTES
            path = member.name.removeprefix(MEMBER_PREFIX)
            if path not in pending:
                continue
            pending.discard(path)
            try:
                img = Image.open(io.BytesIO(tar.extractfile(member).read()))
                img.load()
                out = io.BytesIO()
                img.save(out, format="PNG")
                pages[path] = out.getvalue()
            except Exception as e:
                failures[path] = f"{type(e).__name__}: {e}"
                print(f"decode failure: {path}: {failures[path]}", file=sys.stderr)
            if not pending:
                break
    for path in pending:
        failures[path] = "not in archive"
        print(f"decode failure: {path}: not in archive", file=sys.stderr)
    print(f"archive pass {time.time() - start:.0f} s, {seen / 2**30:.1f} GiB of members read",
          file=sys.stderr)
    return pages, failures


def main():
    names = rvl_label_names()
    if sorted(names) != sorted(list(CLASSES) + REJECT_CLASSES):
        sys.exit(f"{REPO} label names {names} are not CLASSES + REJECT_CLASSES")

    rows, draws = {}, {}
    for split, (rvl_split, _, _) in SPLITS.items():
        rows[split] = label_list(rvl_split)
        draws[split] = draw(split, rows[split], names)
    paths = {rows[s][i][0] for s in SPLITS for *_, cand in draws[s] for i in cand}
    pages, failures = stream_pages(paths)

    manifest, selection, short = [], {}, []
    for split, (rvl_split, _, _) in SPLITS.items():
        os.makedirs(f"data/{split}", exist_ok=True)
        indices = {}
        for label, rvl_label, k, cand in draws[split]:
            kept = [i for i in cand if rows[split][i][0] in pages][:k]
            if len(kept) < k:
                short.append(f"{split} {rvl_label}: {len(kept)} of {k}")
            indices[rvl_label] = kept
            for i in kept:
                page_id = f"{split}-{i:06d}"
                with open(f"data/{split}/{page_id}.png", "wb") as f:
                    f.write(pages[rows[split][i][0]])
                manifest.append({"id": page_id, "split": split, "label": label,
                                 "rvl_label": rvl_label, "rvl_index": i})
        selection[split] = {"rvl_split": rvl_split, "rvl_indices": indices}

    with open("data/manifest.jsonl", "w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")
    with open("data/selection.json", "w") as f:
        json.dump({"seed": SEED, "source": REPO, "spare": SPARE, "splits": selection,
                   "decode_failures": dict(sorted(failures.items()))}, f, indent=1)
        f.write("\n")

    for split in SPLITS:
        n = sum(r["split"] == split for r in manifest)
        rej = sum(r["split"] == split and r["label"] == REJECT_LABEL for r in manifest)
        print(f"{split}: {n} pages, {rej} reject", file=sys.stderr)
    print(f"decode failures: {len(failures)}", file=sys.stderr)
    if short:
        sys.exit("short of pages after spares: " + "; ".join(short))


if __name__ == "__main__":
    main()
