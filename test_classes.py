"""test_classes.py checks classes.py against the corpus: CLASSES are exactly the labels of the
RVL-CDIP-N rows of data/pdf/manifest.jsonl less HANDWRITTEN, so every in-class page has a type to
classify and every type has pages."""

import json

from classes import CLASSES, HANDWRITTEN, REJECT_LABEL


def test_classes_match_manifest_labels():
    labels = manifest_n_labels()
    assert labels, "no N rows in data/pdf/manifest.jsonl"
    assert HANDWRITTEN in labels
    assert sorted(CLASSES) == sorted(labels - {HANDWRITTEN})
    assert REJECT_LABEL not in CLASSES


def manifest_n_labels():
    """The distinct `label` values of the N rows of data/pdf/manifest.jsonl."""
    with open("data/pdf/manifest.jsonl") as f:
        return {r["label"] for r in map(json.loads, f) if r["set"] == "N"}
