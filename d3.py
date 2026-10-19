"""d3.py classifies pages by their OCR text with a Decision 3.0 model from vLLM Semantic Router,
such as `vllm-sr/d3-flash`, run locally through transformers: the model answers a `choice` question
with a probability for every option. The question's criteria are CLASSES' definitions, and with
the `fewshot` prompt each definition is followed by the OCR text of its data/fewshot.txt pages. It
reads nothing but the OCR cache and data/fewshot.txt; score.py passes the page ids in. Copied from
synth-classify's d3.py, with the question and the state built here instead of by jev.py.

The model's code runs with `trust_remote_code` and needs transformers 5.17.0, which is not the
version in pyproject.toml, so it runs in its own venv, `.venv-d3`, made from requirements-d3.txt.

Every response is cached in results/d3/cache.jsonl under the page's id, the prompt and the model
id, so a re-run of score.py reproduces its report, `seconds` line included, without loading the
model."""

import hashlib
import json
import os
import re
import sys
import time

from classes import CLASSES
from train import manifest, page_text

# The prompts `d3_question` builds; the report name is <model basename>-<key>.
D3_PROMPTS = ["definitions", "fewshot"]
# The commit each model's code and weights are loaded at, since the code runs with
# `trust_remote_code`; a model not listed loads at its main branch.
REVISIONS = {"vllm-sr/d3-flash": "13dc1cc8ef23caefa765907e1d764751e5629f7d"}
CACHE = "results/d3/cache.jsonl"
FEWSHOT = "data/fewshot.txt"
# A page's OCR text is cut to its first STATE_WORDS words as the state, and a fewshot example's to
# its first EXAMPLE_WORDS, so the 36 examples and the page fit one prompt. Fixed, not tuned.
STATE_WORDS = 1500
EXAMPLE_WORDS = 300


def first_words(text, n):
    """first_words returns `text` up to the end of its `n`th whitespace-separated word, its line
    breaks kept."""
    for i, m in enumerate(re.finditer(r"\S+", text)):
        if i == n - 1:
            return text[: m.end()]
    return text


def fewshot_ids():
    """fewshot_ids returns {class name: [page id]} from data/fewshot.txt, in file order."""
    with open(FEWSHOT) as f:
        ids = [line.strip() for line in f if line.strip()]
    label = {r["id"]: r["label"] for r in manifest()}
    shots = {name: [] for name in CLASSES}
    for page_id in ids:
        shots[label[page_id]].append(page_id)
    return shots


def d3_question(prompt):
    """d3_question is the `choice` question the model is asked about each page: the CLASSES names as
    options with their definitions as criteria, and for `fewshot` each definition followed by the
    first EXAMPLE_WORDS words of the OCR text of each of that class's data/fewshot.txt pages."""
    criteria = dict(CLASSES)
    if prompt == "fewshot":
        shots = fewshot_ids()
        criteria = {
            name: definition
            + "".join(
                f"\nexample: {first_words(page_text(page_id), EXAMPLE_WORDS)}"
                for page_id in shots[name]
            )
            for name, definition in CLASSES.items()
        }
    return {
        "type": "choice",
        "instructions": "Which type of document is this page, given its OCR text?",
        "criteria": criteria,
    }


def read_cache(path):
    """read_cache returns {(page id, prompt, model id): cached row}; a later row for a key replaces
    an earlier one."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {(r["id"], r["prompt"], r["model"]): r for r in rows}


def classify_d3(ids, prompt, model_id):
    """classify_d3 returns a (class name, probabilities, seconds) tuple for each page of `ids`: the
    option the model chose, every option's probability as a list in CLASSES order, and the wall time
    of that page's `system_one` call. The state is the page's OCR text cut to STATE_WORDS words.
    `prompt` is a D3_PROMPTS key. A cached row is used when it was asked of the same revision, the
    same question and the same state; the model is loaded, on MPS in bfloat16, only if some page is
    not cached."""
    names = list(CLASSES)
    question = d3_question(prompt)
    revision = REVISIONS.get(model_id)
    cache = read_cache(CACHE)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    model = None
    out = []
    for i, page_id in enumerate(ids):
        state = first_words(page_text(page_id), STATE_WORDS)
        digest = hashlib.sha256(state.encode()).hexdigest()
        row = cache.get((page_id, prompt, model_id))
        if (
            row is None
            or row["revision"] != revision
            or row["question"] != question
            or row["state_sha256"] != digest
        ):
            if model is None:
                model = load(model_id, revision)
            start = time.perf_counter()
            response = model.system_one(state=state, questions={"type": question})
            seconds = time.perf_counter() - start
            row = {
                "id": page_id,
                "prompt": prompt,
                "model": model_id,
                "revision": revision,
                "question": question,
                "state_sha256": digest,
                "seconds": seconds,
                "response": response,
            }
            # Appended per page, so an interrupted run resumes where it stopped.
            with open(CACHE, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{i + 1}/{len(ids)} {page_id} ({seconds:.2f} s)", file=sys.stderr)
        answer = row["response"]["answers"]["type"]
        if "error" in answer:
            sys.exit(f"{page_id}: {model_id} answered {answer['error']!r}")
        if answer["choice"] not in names:
            raise ValueError(f"{page_id}: {model_id} chose {answer['choice']!r}, not in CLASSES")
        vector = [float(answer["probabilities"][name]) for name in names]
        out.append((answer["choice"], vector, row["seconds"]))
    return out


def load(model_id, revision):
    """load returns the model on MPS in bfloat16, and exits rather than fall back to the CPU."""
    import torch
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoModel

    if not torch.backends.mps.is_available():
        sys.exit("d3.py: MPS is not available; it does not fall back to the CPU")
    print("device: mps", file=sys.stderr)
    # The model's runtime calls snapshot_download for a Hub id, which lists the repository even
    # under HF_HUB_OFFLINE; a cached snapshot is passed as its directory, which it reads as is.
    cached = try_to_load_from_cache(model_id, "config.json", revision=revision)
    source = os.path.dirname(cached) if isinstance(cached, str) else model_id
    return AutoModel.from_pretrained(
        source, trust_remote_code=True, revision=revision, device="mps", dtype=torch.bfloat16
    )
