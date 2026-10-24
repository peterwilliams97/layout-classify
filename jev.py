"""jev.py classifies pages by their OCR text with TypeSafe's System One API ("Jev"), which answers a
`choice` question with a probability for every option, so a hosted model can be set beside d3.py's
local d3-flash on the same question. Each page is one POST /v1/systemone request: the page's OCR
text cut to its first STATE_WORDS words as `state`, and d3.py's `d3_question` for `definitions`
with REJECT_LABEL an option as the one question. The question is imported, not copied, so the two
cannot drift apart. Copied from synth-classify's jev.py, with the question and the state built by
d3.py instead of llm.py. It reads nothing but the OCR cache; score.py passes the page ids in.

The pages leave the machine: the request goes to OpenRouter, and a page's seconds are the request's
wall time, network round trip included.

Every response is cached in results/jev/cache.jsonl under the page's id and the option names, so a
re-run of score.py reproduces its report, `seconds` line included, with no call. The key comes
from `TYPESAFE_API_KEY` and is never written anywhere. `TYPESAFE_BASE_URL` and `TYPESAFE_MODEL`
override `BASE_URL` and `MODEL`."""

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

from d3 import EXAMPLE_WORDS, d3_question, first_words
from train import page_text

# System One through OpenRouter, as in synth-classify's jev.py.
BASE_URL = "https://openrouter.ai/api"
MODEL = "typesafe/jev-1.13"
CACHE = "results/jev/cache.jsonl"
# The state is the page's first STATE_WORDS words, d3-flash's `--state-words 150`.
STATE_WORDS = 150


def jev_question():
    """jev_question is d3.py's `d3_question` for `definitions` over all 11 CLASSES with
    REJECT_LABEL a 12th option, the question of score.py's
    `--d3 vllm-sr/d3-flash --state-words 150 --reject-option`."""
    return d3_question("definitions", EXAMPLE_WORDS, True, "all")


def read_cache(path):
    """read_cache returns {(page id, option names): cached row}; a later row for a key replaces an
    earlier one."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {(r["id"], tuple(r["options"])): r for r in rows}


def request_body(model, state, question):
    """request_body is the JSON body of one /v1/systemone request; the key goes in a header."""
    return {"model": model, "state": state, "questions": {"type": question}}


def classify_jev(ids):
    """classify_jev returns a (class name, probabilities, seconds) tuple for each page of `ids`: the
    option System One chose, every option's probability as a list in `jev_question`'s option order,
    REJECT_LABEL last, and the wall time of that page's request. A cached row is used when it was
    asked of the same model, the same question and the same state; otherwise the page is sent and
    the response appended to the cache. A failed request exits, with no retry."""
    question = jev_question()
    names = list(question["criteria"])
    model = os.environ.get("TYPESAFE_MODEL", MODEL)
    url = os.environ.get("TYPESAFE_BASE_URL", BASE_URL).rstrip("/") + "/v1/systemone"
    cache = read_cache(CACHE)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    key = None
    out = []
    for i, page_id in enumerate(ids):
        state = first_words(page_text(page_id), STATE_WORDS)
        digest = hashlib.sha256(state.encode()).hexdigest()
        row = cache.get((page_id, tuple(names)))
        if (
            row is None
            or row["requested"] != model
            or row["question"] != question
            or row["state_sha256"] != digest
        ):
            if key is None:
                key = os.environ.get("TYPESAFE_API_KEY")
                if not key:
                    sys.exit("TYPESAFE_API_KEY is not set")
            req = urllib.request.Request(
                url,
                data=json.dumps(request_body(model, state, question)).encode(),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            )
            start = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    response = json.load(resp)
            except urllib.error.HTTPError as e:
                sys.exit(f"{url}: HTTP {e.code} on {page_id}: {e.read().decode(errors='replace')}")
            except (urllib.error.URLError, TimeoutError) as e:
                sys.exit(f"{url}: {e} on {page_id}")
            seconds = time.perf_counter() - start
            row = {
                "id": page_id,
                "options": names,
                "requested": model,
                "question": question,
                "state_words": STATE_WORDS,
                "state_sha256": digest,
                "seconds": seconds,
                "response": response,
            }
            # Appended per page, so an interrupted run resumes where it stopped.
            with open(CACHE, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{i + 1}/{len(ids)} {page_id} ({seconds:.2f} s)", file=sys.stderr)
        answer = row["response"].get("answers", {}).get("type")
        if answer is None or "error" in answer:
            sys.exit(f"{page_id}: {model} gave no answer: {json.dumps(row['response'])[:500]}")
        if answer["choice"] not in names:
            sys.exit(f"{page_id}: {model} chose {answer['choice']!r}, not an option")
        vector = [float(answer["probabilities"][name]) for name in names]
        out.append((answer["choice"], vector, row["seconds"]))
    return out
