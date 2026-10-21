"""test_confidence.py checks confidence.py's cut-offs and test figures on a hand-made 13-row dump
whose right answers were worked out by hand: 10 in-class pages and 3 reject pages. One reject page
is given a type and scores above the alpha 0.10 cut-off, and so counts as an accepted error and as
reject leak; one is predicted REJECT_LABEL with the top score, which is right and never accepted."""

import json
import sys

import numpy as np

from confidence import draw, main, read_dump, report, threshold
from classes import CLASSES, REJECT_LABEL


def test_cut_offs_count_a_reject_page_as_wrong(tmp_path):
    path = write_dump(tmp_path / "dump.jsonl")
    score, correct, _ = read_dump(path, NAMES)
    assert correct.sum() == 10
    assert np.isinf(score[0]) and score[0] < 0
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
    # At tau 0.70: 9 of 10 in-class pages, all right; 1 of 3 reject pages, the one predicted
    # REJECT_LABEL not among them; 10 of 13 pages, 1 wrong.
    assert "    tau    0.700000\n" in alpha_10
    assert "      in-class             coverage 0.9000 error 0.0000\n" in alpha_10
    assert "      reject_leak          0.3333 (1 of 3)\n" in alpha_10
    assert "      all pages, 3 of 13 reject  coverage 0.7692 error 0.1000\n" in alpha_10
    # tau_cp accepts nothing.
    assert "      in-class             coverage 0.0000 error -\n" in alpha_10
    assert "      reject_leak          0.0000 (0 of 3)\n" in alpha_10
    alpha_05 = text[text.index("\ntest: "):].split("  alpha 0.05\n")[1]
    # At tau 0.93: the top 4 scored pages, all in-class and right.
    assert "      in-class             coverage 0.4000 error 0.0000\n" in alpha_05
    assert "      all pages, 3 of 13 reject  coverage 0.3077 error 0.0000\n" in alpha_05
    assert "0 of 10 in-class pages, 1 of 3 reject pages" in text


def test_report_when_every_scored_in_class_page_is_right(tmp_path):
    # ROWS with its specification page right and a budget page predicted REJECT_LABEL: the one
    # in-class error has no score, so AUROC and the mean wrong score are undefined on test.
    path = tmp_path / "dump.jsonl"
    rows = ROWS[:11] + [(0.60, "specification", "specification"), ROWS[12]]
    write_dump(path, rows + [(0.40, "budget", REJECT_LABEL)])
    text = report("hand", str(path), str(path))
    assert "  AUROC of score against correctness, in-class  -\n" in text
    assert "  mean score, wrong predictions       - (0)\n" in text
    assert "1 of 11 in-class pages, 1 of 3 reject pages" in text


def test_calibration_pool_fits_on_the_pool_dump(tmp_path, monkeypatch):
    # The pool dump is the hand-made dump; the test dump is the same rows with the reject page at
    # 0.91 predicted REJECT_LABEL, so only the pool dump puts a reject page above 0.70.
    dumps = tmp_path / "results" / "confidence" / "dumps"
    dumps.mkdir(parents=True)
    write_dump(dumps / "hand-pool.jsonl")
    test_rows = ROWS[:5] + [(0.91, REJECT_LABEL, REJECT_LABEL)] + ROWS[6:]
    write_dump(dumps / "hand-test.jsonl", test_rows)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["confidence.py", "--model", "hand", "--calibration", "pool"])
    main()
    text = (tmp_path / "results" / "confidence" / "hand-pool.txt").read_text()
    assert text.startswith("model     hand\npool results/confidence/dumps/hand-pool.jsonl (13 pages,")
    # The cut-offs are the pool dump's, as test_cut_offs_count_a_reject_page_as_wrong works out.
    assert "    tau    0.700000  pool coverage 0.7692 error 0.1000 reject_leak 0.3333\n" in text
    alpha_10 = text[text.index("\ntest: "):].split("  alpha 0.10\n")[1]
    # On test no reject page is given a type, so none is accepted at tau 0.70: 9 of 13 pages, all
    # in-class and right.
    assert "    tau    0.700000\n" in alpha_10
    assert "      reject_leak          0.0000 (0 of 3)\n" in alpha_10
    assert "      all pages, 3 of 13 reject  coverage 0.6923 error 0.0000\n" in alpha_10


def test_calibration_pool_averages_the_fold_dumps(tmp_path, monkeypatch):
    # Two fold models' test dumps: fold0 is the hand-made dump, fold1 the same rows with the reject
    # page at 0.91 predicted REJECT_LABEL. The pool cut-offs are the hand-made dump's.
    dumps = tmp_path / "results" / "confidence" / "dumps"
    dumps.mkdir(parents=True)
    write_dump(dumps / "hand-pool.jsonl")
    write_dump(dumps / "hand-fold0-test.jsonl")
    fold1 = ROWS[:5] + [(0.91, REJECT_LABEL, REJECT_LABEL)] + ROWS[6:]
    write_dump(dumps / "hand-fold1-test.jsonl", fold1)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["confidence.py", "--model", "hand", "--calibration", "pool"])
    main()
    text = (tmp_path / "results" / "confidence" / "hand-pool.txt").read_text()
    assert text.startswith("model     hand, 2 fold models\n")
    summary = text.split("\n\n")[0].split("  alpha 0.10\n")[1]
    # At tau 0.70, fold0 accepts 10 of 13 pages with 1 wrong, the leaked reject page; fold1 accepts
    # 9, all right. tau_cp accepts nothing in either.
    assert "    tau    0.700000\n" in summary
    assert "      in-class    coverage 0.9000 (0.9000–0.9000)\n" in summary
    assert "      reject_leak          0.1667 (0.0000–0.3333), 1/0 of 3 by fold\n" in summary
    assert "      all pages   coverage 0.7308 (0.6923–0.7692)\n" in summary
    assert "      all pages   error    0.0500 (0.0000–0.1000)\n" in summary
    assert summary.endswith("      all pages   error    -      (no fold accepts a page)")
    # Each fold's full report follows the summary.
    assert "test: results/confidence/dumps/hand-fold0-test.jsonl (13 pages, 3 reject)" in text
    assert "test: results/confidence/dumps/hand-fold1-test.jsonl (13 pages, 3 reject)" in text


def test_draw_takes_one_page_per_class_present(tmp_path):
    _, _, label = read_dump(write_dump(tmp_path / "dump.jsonl"), NAMES)
    picked = draw(label, len(NAMES), 0, len(NAMES))
    # 10 of the 11 classes and the reject pages are present; scientific_publication is not.
    assert len(picked) == 11
    assert (label[picked] == NAMES.index(REJECT_LABEL)).sum() == 1
    assert list(picked) == list(draw(label, len(NAMES), 0, len(NAMES)))


NAMES = list(CLASSES) + [REJECT_LABEL]

# (score, label, predicted), highest score first.
ROWS = [
    (0.98, REJECT_LABEL, REJECT_LABEL),
    (0.99, "letter", "letter"),
    (0.97, "memo", "memo"),
    (0.95, "form", "form"),
    (0.93, "email", "email"),
    (0.91, REJECT_LABEL, "letter"),
    (0.89, "budget", "budget"),
    (0.85, "invoice", "invoice"),
    (0.80, "resume", "resume"),
    (0.75, "questionnaire", "questionnaire"),
    (0.70, "news_article", "news_article"),
    (0.60, "specification", "form"),
    (0.50, REJECT_LABEL, "memo"),
]


def write_dump(path, rows=ROWS):
    """Writes `rows` as a score.py --dump file at `path` and returns the path as a string."""
    with open(path, "w") as f:
        for i, (score, label, predicted) in enumerate(rows):
            row = {"id": f"p{i}", "label": label, "predicted": predicted, "score": score}
            f.write(json.dumps(row) + "\n")
    return str(path)
