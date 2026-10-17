"""test_classes.py checks classes.py against the published dataset: CLASSES and REJECT_CLASSES
together are exactly the 16 label names of `aharley/rvl_cdip`, each once, so every RVL-CDIP page
is either a type to classify or a reject page."""

from classes import CLASSES, REJECT_CLASSES
from data import rvl_label_names


def test_classes_cover_rvl_cdip_labels():
    names = rvl_label_names()
    assert len(names) == 16
    assert sorted(list(CLASSES) + REJECT_CLASSES) == sorted(names)
