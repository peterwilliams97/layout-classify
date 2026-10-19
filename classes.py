"""CLASSES holds the 12 RVL-CDIP-N document types a classifier here must tell apart, each with the
one-sentence definition that is the only description of the type an LLM or VLM prompt gets. Each
definition says what separates the type from the one it is most easily taken for, above all in the
three near-pairs `letter` / `memo`, `form` / `questionnaire` and `budget` / `invoice`. Names are the
`label` strings of the N rows of data/pdf/manifest.jsonl, which fetch.py takes from the RVL-CDIP-N
PDF file names; test_classes.py checks the two agree.

Reject pages come from RVL-CDIP-O PDFs, documents of none of these types, and are labelled
REJECT_LABEL."""

CLASSES = {
    "letter":                 "A letter addressed to a named recipient, opening with a salutation and closing with a signature, often on letterhead, where a memo has a To / From / Subject header instead.",
    "memo":                   "An internal memorandum headed by To, From, Date and Subject fields, with neither the salutation nor the closing signature of a letter.",
    "form":                   "A page of labelled fields, boxes or blanks to be filled in with names, dates, amounts or other particulars, where a questionnaire asks numbered questions for a respondent's answers or opinions.",
    "questionnaire":          "A survey of numbered questions put to a respondent, with answer choices, scales or space to reply, where a form collects administrative particulars in labelled fields.",
    "budget":                 "A table of planned or actual amounts of money by category or period for an organisation or project, billing no one, where an invoice asks a customer to pay.",
    "invoice":                "A bill from a supplier to a named customer, with an invoice number and date, the goods or services, their prices and a total due, where a budget plans or reports spending.",
    "email":                  "A printed email message with From, To, Sent or Date and Subject header lines and email addresses, where a memo's header names people and departments without addresses or a send time.",
    "news_article":           "An article from a newspaper or magazine under a headline, typeset in columns, often with a byline and dateline, where a scientific publication has authors' affiliations, an abstract and references.",
    "scientific_publication": "A journal or conference article with a title, authors and their affiliations, an abstract and references, where a news article has a headline and byline for a general reader.",
    "specification":          "A technical specification setting out the required properties, dimensions or tests of a product, material or process, in numbered clauses or tables, where a form has fields left blank to fill in.",
    "resume":                 "A person's curriculum vitae listing their contact details, education, work history and skills, where a letter is addressed to a recipient.",
    "handwritten":            "A page whose content is mostly handwriting rather than print, of any kind of document, where the other types are mostly printed or typed.",
}

REJECT_LABEL = "none"
