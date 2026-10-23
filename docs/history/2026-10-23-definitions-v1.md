# Definitions v1, at `91c3c90`

The three sentences below are how the prompt read at `91c3c90` (2026-10-23), before `REJECT_CRITERION`
listed the kinds of reject page and the `specification` and `news_article` definitions were extended.
`d3-flash-definitions-s150-reject`'s pool results from that prompt are in `results/definitions-v1/`:
`pool.txt` (was `results/pool/d3-flash-definitions-s150-reject.txt`), `confidence-pool.txt` (was
`results/confidence/d3-flash-definitions-s150-reject-pool.txt`) and `dump-pool.jsonl` (was
`results/confidence/dumps/d3-flash-definitions-s150-reject-pool.jsonl`), and its test results are
`test.txt` (was `results/test/d3-flash-definitions-s150-reject.txt`) and `dump-test.jsonl` (was
`results/confidence/dumps/d3-flash-definitions-s150-reject-test.jsonl`). `d3-flash-definitions-s150`'s,
the same prompt without `none` as an option, carry a `without-none-` prefix: `without-none-pool.txt`,
`without-none-dump-pool.jsonl`, `without-none-test.txt`, `without-none-dump-test.jsonl` and
`without-none-confidence-pool.txt`, from the same places under that name. Validation results and
the `d3-flash-fewshot-s150-e50` results were not rescored and are still from this prompt.

`d3.py` `REJECT_CRITERION`:

> A page that fits none of the 11 types.

`classes.py` `specification`:

> A technical specification setting out the required properties, dimensions or tests of a product,
> material or process, in numbered clauses or tables, where a form has fields left blank to fill in.

`classes.py` `news_article`:

> An article from a newspaper or magazine under a headline, typeset in columns, often with a byline
> and dateline, where a scientific publication has authors' affiliations, an abstract and
> references.
