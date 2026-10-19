"""llm.py classifies pages with a local vision LLM, gemma4:e4b on Ollama, given the page image and
nothing else: no OCR text and no hand-labelled page. `PROMPT` and `REJECT_PROMPT` are the
instructions, built from the CLASSES definitions in classes.py, and `classify` sends them with one
page image at a time and gets back one type name. The reply is restricted to the type names by
Ollama's structured output, so there is no probability and no score to set a cut-off on. Adapted
from synth-classify's llm.py."""

import base64
import json
import os
import time
import urllib.request

from classes import CLASSES, REJECT_LABEL

MODEL = "gemma4:e4b"

INTRO = (
    "You sort scanned or printed document pages by type. "
    "Answer with the one type that fits the page in the image best."
)

PROMPT = INTRO + "\n\nTypes:\n" + "\n".join(f"- {n}: {d}" for n, d in CLASSES.items())

# REJECT_PROMPT is PROMPT with REJECT_LABEL as one more answer, for a page of none of the types.
REJECT_PROMPT = PROMPT + f'\n\nIf the page is none of these types, answer "{REJECT_LABEL}".'

# score.py's `--llm` names, each mapped to `classify`'s `allow_reject`.
NAMES = {"gemma4-image": False, "gemma4-image-reject": True}


def classify(image_path, allow_reject, url="http://localhost:11434"):
    """classify returns a (type name, seconds) pair for the page image at `image_path`. The answer
    is a name in CLASSES, or with `allow_reject` also REJECT_LABEL. Decoding is temperature 0 with
    thinking off. The seconds are the wall time of the request, the file read and base64 encoding
    excluded, so the first page's include Ollama loading `MODEL`."""
    with open(image_path, "rb") as f:
        image = base64.b64encode(f.read()).decode()
    names = list(CLASSES) + ([REJECT_LABEL] if allow_reject else [])
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": REJECT_PROMPT if allow_reject else PROMPT},
            {"role": "user", "content": "Page:", "images": [image]},
        ],
        "format": {
            "type": "object",
            "properties": {"type": {"type": "string", "enum": names}},
            "required": ["type"],
        },
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "seed": 0},
    }
    req = urllib.request.Request(
        url + "/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        content = json.load(resp)["message"]["content"]
    seconds = time.perf_counter() - start
    label = json.loads(content)["type"]
    if label not in names:
        raise ValueError(f"{MODEL} answered {label!r} for {os.path.basename(image_path)}")
    return label, seconds

