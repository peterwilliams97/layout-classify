"""CLASSES holds the 10 RVL-CDIP document types a classifier here must tell apart, each with the
one-sentence definition that is the only description of the type an LLM or VLM prompt gets. They
include four near-pairs: `letter` / `memo`, `form` / `questionnaire`, `budget` / `invoice` and
`scientific report` / `scientific publication`.

REJECT_CLASSES are the other 6 RVL-CDIP types. Their pages appear in dev and test only, labelled
REJECT_LABEL, as documents that fit none of CLASSES. Names are RVL-CDIP's own label names, as in
the `label` feature of `aharley/rvl_cdip`; test_classes.py checks that the two lists cover them."""

CLASSES = {
    "letter":                 "A letter from one person or organisation to another, with a salutation, body and signature, usually on letterhead.",
    "memo":                   "An internal memorandum with To, From, Date and Subject header fields and no salutation or letterhead address block.",
    "form":                   "A document of labelled fields or boxes to be filled in with administrative or transactional details.",
    "questionnaire":          "A set of numbered questions put to a respondent, with answer choices or space for answers.",
    "budget":                 "A table of planned or actual amounts of money broken down by category or period.",
    "invoice":                "A bill from a supplier to a customer listing goods or services, their prices and the amount due.",
    "scientific report":      "A technical or research report on an experiment, study or test, written for an organisation rather than a journal.",
    "scientific publication": "A journal or conference article with a title, authors, abstract and references, typeset in columns.",
    "email":                  "A printed email message with From, To, Sent and Subject header lines.",
    "news article":           "An article from a newspaper or magazine, typeset in columns under a headline.",
}

REJECT_CLASSES = [
    "handwritten",
    "advertisement",
    "specification",
    "file folder",
    "presentation",
    "resume",
]

REJECT_LABEL = "none of these"
