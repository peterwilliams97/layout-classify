"""test_classes.py checks classes.py against the published dataset: CLASSES and REJECT_CLASSES
together are exactly the 16 label names of `aharley/rvl_cdip`, each once, so every RVL-CDIP page
is either a type to classify or a reject page."""

import json

from huggingface_hub import hf_hub_download

from classes import CLASSES, REJECT_CLASSES


def test_classes_cover_rvl_cdip_labels():
    names = rvl_label_names()
    assert len(names) == 16
    assert sorted(list(CLASSES) + REJECT_CLASSES) == sorted(names)


def rvl_label_names():
    """The 16 RVL-CDIP label names in `aharley/rvl_cdip`'s class-id order, read from its
    dataset_infos.json."""
    path = hf_hub_download("aharley/rvl_cdip", "dataset_infos.json", repo_type="dataset")
    with open(path) as f:
        return json.load(f)["default"]["features"]["label"]["names"]
