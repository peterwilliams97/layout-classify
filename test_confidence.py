"""test_confidence.py checks confidence.py's cut-offs and test figures on a hand-made 12-row dump
whose right answers were worked out by hand: 10 in-class pages and 2 reject pages, one of which
scores above the alpha 0.10 cut-off and so counts as an accepted error and as reject leak."""

import json

import numpy as np

from confidence import draw, read_dump, report, threshold
from classes import CLASSES, REJECT_LABEL


def test_cut_offs_count_a_reject_page_as_wrong(tmp_path):
    path = write_dump(tmp_path / "dump.jsonl")
    score, correct, _ = read_dump(path, NAMES)
    assert correct.sum() == 9
    # Errors by depth: none in the top 4, the reject page at 5, then 1 in 10 down to 0.70.
    assert threshold(score, correct, 0.10, bound=False) == 0.70
    assert threshold(score, correct, 0.05, bound=False) == 0.93
    assert threshold(score, correct, 0.02, bound=False) == 0.93
    # 4 right in a row bound the error at 1 - 0.1 ** (1 / 4) = 0.44, above every alpha.
    for alpha in (0.02, 0.05, 0.10):
        assert np.isinf(threshold(score, correct, alpha, bound=True))


def test_report_on_test_pages(tmp_path):
    path = write_dump(tmp_path / "dump.jsonl")
    text = report("hand", path, path)
    alpha_10 = text[text.index("\ntest: "):].split("  alpha 0.10\n")[1]
    # At tau 0.70: 9 of 10 in-class pages, all right; 1 of 2 reject pages; 10 of 12 pages, 1 wrong.
    assert "    tau    0.700000\n" in alpha_10
    assert "      in-class             coverage 0.9000 error 0.0000\n" in alpha_10
    assert "      reject_leak          0.5000 (1 of 2)\n" in alpha_10
    assert "      all pages, 2 of 12 reject  coverage 0.8333 error 0.1000\n" in alpha_10
    # tau_cp accepts nothing.
    assert "      in-class             coverage 0.0000 error -\n" in alpha_10
    assert "      reject_leak          0.0000 (0 of 2)\n" in alpha_10
    alpha_05 = text[text.index("\ntest: "):].split("  alpha 0.05\n")[1]
    # At tau 0.93: the top 4 pages, all in-class and right.
    assert "      in-class             coverage 0.4000 error 0.0000\n" in alpha_05
    assert "      all pages, 2 of 12 reject  coverage 0.3333 error 0.0000\n" in alpha_05


def test_draw_takes_one_page_per_class_present(tmp_path):
    _, _, label = read_dump(write_dump(tmp_path / "dump.jsonl"), NAMES)
    picked = draw(label, len(NAMES), 0, len(NAMES))
    # 10 classes and the reject pages are present; questionnaire and scientific_publication are not.
    assert len(picked) == 11
    assert (label[picked] == NAMES.index(REJECT_LABEL)).sum() == 1
    assert list(picked) == list(draw(label, len(NAMES), 0, len(NAMES)))


NAMES = list(CLASSES) + [REJECT_LABEL]

# (score, label, predicted), highest score first.
ROWS = [
    (0.99, "letter", "letter"),
    (0.97, "memo", "memo"),
    (0.95, "form", "form"),
    (0.93, "email", "email"),
    (0.91, REJECT_LABEL, "letter"),
    (0.89, "budget", "budget"),
    (0.85, "invoice", "invoice"),
    (0.80, "resume", "resume"),
    (0.75, "handwritten", "handwritten"),
    (0.70, "news_article", "news_article"),
    (0.60, "specification", "form"),
    (0.50, REJECT_LABEL, "memo"),
]


def write_dump(path):
    """Writes ROWS as a score.py --dump file at `path` and returns the path as a string."""
    with open(path, "w") as f:
        for i, (score, label, predicted) in enumerate(ROWS):
            row = {"id": f"p{i}", "label": label, "predicted": predicted, "score": score}
            f.write(json.dumps(row) + "\n")
    return str(path)
