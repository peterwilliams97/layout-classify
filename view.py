"""view.py turns classify.py's CSV into one static HTML page for a person reading the results: a row
per file with its page-1 thumbnail, type, score, decision and seconds, filtered by decision and by
type in the browser. The page has its CSS and JS inline and fetches nothing; the thumbnails are
files beside the folder's documents, linked by path relative to the HTML. With --embed each
thumbnail is instead a THUMB_WIDTH-pixel JPEG inside the page as a data URI, so the HTML is one file
that can be sent on its own; the thumbnail then links nowhere, as the full page is not in it.

A PDF's page 1 is rendered by render.py's `render` at its DPI to
<folder>/.layout-classify/thumbs/<sha256 of the file>.png, kept so a second run renders nothing; an
image file is linked as it is.

A file's label comes from --labels, a CSV with columns `file` and `label`, if it is there; else from
the data/manifest.jsonl row whose `id` is the file name less its extension, as for the pages of
data/<split>/; else it has none. The row is light blue when `type` equals the label, light orange
when it differs, and plain without a label. The label `none` is a label, so a reject page
classified `none` is blue.

  .venv-d3/bin/python view.py results.csv <folder> [--labels labels.csv] [--embed] [--out results.html]"""

import argparse
import base64
import csv
import hashlib
import html
import io
import json
import os
import sys

from PIL import Image

from render import render

MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "manifest.jsonl")
CACHE_DIR = ".layout-classify"
THUMB_WIDTH = 200


def main():
    ap = argparse.ArgumentParser(description="Write an HTML page of classify.py's CSV.")
    ap.add_argument("csv")
    ap.add_argument("folder", help="the folder classify.py was run on")
    ap.add_argument("--labels", help="CSV of file,label; read before data/manifest.jsonl")
    ap.add_argument("--embed", action="store_true",
                    help="put the thumbnails in the HTML so it is one file to send")
    ap.add_argument("--out", default="results.html", help="HTML written; default results.html")
    args = ap.parse_args()

    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{args.csv}: no rows")
    given = {}
    if args.labels:
        with open(args.labels, newline="") as f:
            given = {r["file"]: r["label"] for r in csv.DictReader(f)}
    manifest = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            manifest = {r["id"]: r["label"] for r in map(json.loads, f)}
    thumb_dir = os.path.join(args.folder, CACHE_DIR, "thumbs")
    out_dir = os.path.dirname(os.path.abspath(args.out))

    for r in rows:
        path = os.path.join(args.folder, r["file"])
        image = path
        if path.lower().endswith(".pdf"):
            image = pdf_thumb(path, thumb_dir)
        if not image:
            r["image"] = ""
        elif args.embed:
            r["image"] = embedded(image)
        else:
            r["image"] = os.path.relpath(image, out_dir)
        r["label"], r["from"] = None, "none"
        if r["file"] in given:
            r["label"], r["from"] = given[r["file"]], "labels"
        elif os.path.splitext(r["file"])[0] in manifest:
            r["label"], r["from"] = manifest[os.path.splitext(r["file"])[0]], "manifest"
        r["state"] = ("nolabel" if r["label"] is None
                      else "agree" if r["type"] == r["label"] else "disagree")

    with open(args.out, "w") as f:
        f.write(page(rows, args.csv, args.labels))
    n = {s: sum(r["state"] == s for r in rows) for s in ("agree", "disagree", "nolabel")}
    print(f"rows {len(rows)}, agree {n['agree']}, disagree {n['disagree']}, "
          f"no label {n['nolabel']}; {args.out}")


def pdf_thumb(path, thumb_dir):
    """pdf_thumb returns <thumb_dir>/<sha256>.png, `path`'s page 1, rendering it if it is missing,
    or "" if the PDF cannot be read, so one bad file leaves its row without a thumbnail."""
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    out = os.path.join(thumb_dir, digest + ".png")
    if not os.path.exists(out):
        os.makedirs(thumb_dir, exist_ok=True)
        try:
            render(path, 0, out)
        except Exception as e:
            print(f"{os.path.basename(path)}: {e}", file=sys.stderr)
            return ""
    return out


def embedded(path):
    """embedded returns `path` shrunk to THUMB_WIDTH pixels wide as a JPEG data URI, or "" if the
    image cannot be read, so one bad file leaves its row without a thumbnail."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((THUMB_WIDTH, THUMB_WIDTH * 4))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=80)
    except Exception as e:
        print(f"{os.path.basename(path)}: {e}", file=sys.stderr)
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def page(rows, source, labels_path):
    """page returns the HTML document for `rows`; the `label` column appears only when some row
    has a label. `labels_path` is the --labels file, named in the header's count of label sources."""
    has_label = any(r["label"] is not None for r in rows)
    types = sorted({r["type"] for r in rows})
    count = {k: sum(r["decision"] == k or r["state"] == k for r in rows)
             for k in ("file", "review", "disagree", "nolabel")}
    sources = {k: sum(r["from"] == k for r in rows) for k in ("labels", "manifest", "none")}
    e = html.escape

    body = []
    for r in rows:
        src = e(r["image"])
        thumb = f'<img src="{src}" width="{THUMB_WIDTH}" loading="lazy" alt="">' if src else ""
        if src and not src.startswith("data:"):
            thumb = f'<a href="{src}">{thumb}</a>'
        cells = [thumb, e(r["file"]), e(r["type"] or "—")]
        if has_label:
            cells.append(e(r["label"] or "—"))
        cells += [e(r["score"]), e(r["decision"]), e(r["seconds"])]
        body.append(f'<tr data-decision="{e(r["decision"])}" data-state="{r["state"]}"'
                    f' data-type="{e(r["type"])}" class="{r["state"]}">'
                    + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    heads = ["page", "file", "type"] + (["label"] if has_label else []) + \
            ["score", "decision", "seconds"]
    options = "".join(
        f'<option value="{e(t)}">{e(t or "(none given)")} ({sum(r["type"] == t for r in rows)})'
        "</option>" for t in types)
    return TEMPLATE.format(
        title=e(os.path.basename(source)),
        labels_name=e(os.path.basename(labels_path) if labels_path else "--labels not given"),
        n_labels=sources["labels"], n_manifest=sources["manifest"], n_none=sources["none"],
        all=len(rows), file=count["file"], review=count["review"],
        disagree=count["disagree"], nolabel=count["nolabel"],
        options=options,
        heads="".join(f"<th>{h}</th>" for h in heads),
        rows="\n".join(body),
    )


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body {{ font: 14px system-ui, sans-serif; margin: 16px; background: #fff; color: #111; }}
.bar {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }}
button {{ font: inherit; padding: 4px 10px; border: 1px solid #888; background: #f4f4f4;
  border-radius: 4px; cursor: pointer; }}
button.on {{ background: #224; color: #fff; border-color: #224; }}
select {{ font: inherit; padding: 3px; }}
table {{ border-collapse: collapse; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 6px 10px; text-align: left; vertical-align: top; }}
th {{ position: sticky; top: 0; background: #fff; }}
tr.agree td, .key.agree {{ background: #dbeafe; }}
tr.disagree td, .key.disagree {{ background: #fde3c4; }}
.key {{ padding: 1px 6px; border: 1px solid #ccc; }}
.note {{ margin: 0 0 8px; color: #333; }}
img {{ display: block; border: 1px solid #ccc; }}
#shown {{ color: #555; }}
</style>
</head>
<body>
<p class="note"><b>{title}</b> · labels: {labels_name} ({n_labels} files), manifest ({n_manifest} files),
none ({n_none} files)</p>
<p class="note"><span class="key agree">label equals type</span>
<span class="key disagree">label differs from type</span>
<span class="key">no label</span></p>
<div class="bar">
<button data-d="" class="on">all ({all})</button>
<button data-d="file">file ({file})</button>
<button data-d="review">review ({review})</button>
<button data-s="disagree">disagree ({disagree})</button>
<button data-s="nolabel">no label ({nolabel})</button>
<select id="type"><option value="*">every type ({all})</option>{options}</select>
<span id="shown"></span>
</div>
<table>
<thead><tr>{heads}</tr></thead>
<tbody>
{rows}
</tbody>
</table>
<script>
let decision = "", state = "";
const type = document.getElementById("type");
const rows = document.querySelectorAll("tbody tr");
function apply() {{
  let n = 0;
  for (const r of rows) {{
    const show = (!decision || r.dataset.decision === decision) &&
      (!state || r.dataset.state === state) &&
      (type.value === "*" || r.dataset.type === type.value);
    r.hidden = !show;
    n += show;
  }}
  document.getElementById("shown").textContent = n + " of " + rows.length + " shown";
}}
for (const b of document.querySelectorAll("button")) {{
  b.onclick = () => {{
    decision = b.dataset.d || "";
    state = b.dataset.s || "";
    for (const o of document.querySelectorAll("button")) o.classList.toggle("on", o === b);
    apply();
  }};
}}
type.onchange = apply;
apply();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
