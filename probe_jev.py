"""The same SDF task, the same gold, run against the real Jev API.

Everything in this repository is a local reproduction of the mechanism TypeSafe's
Jev exposes: give a model state once, ask several typed questions over it, read
typed answers with calibrated probabilities instead of parsing generated text.
This runs the real thing on exactly the task probe_sdf.py scores locally, against
exactly the same URL-derived gold labels, so the comparison is like for like.

The question set is the conditional type path: one Choice for the SDF parent
category, then one Choice per category for its subtype, all in a single request.
TypeSafe calls this speculative fan-out, and the premise of each speculative
question is stated in its own instructions, as the docs require. Code takes the
subtype belonging to the chosen parent, so the path cannot be inconsistent.

The API key is read from JEV_API_KEY, in the environment or in a .env file. It is
never printed.

    python probe_jev.py [n] [dev|test]
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from collections import Counter

import numpy as np

from probe_sdf import PARENTS, SUBTYPES, TAXONOMY, split

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"

# One line per category, lifted from the same taxonomy the local reader is given,
# so neither side gets a better description of the task than the other.
CRITERIA = {
    "article": "Journalism or a written piece published at a point in time: news, "
               "blog post, opinion column, review, press release. A news story "
               "about science is still an article.",
    "documentation": "Material explaining how to use a specific product or "
                     "technology: API reference, guide, tutorial, support page, "
                     "changelog, FAQ. Published by the maker of the thing.",
    "commerce": "A page whose purpose is a transaction: a product for sale, a job "
                "posting, a property listing, a service offering, an auction.",
    "data": "A page whose substance is figures: market data, sports results, "
            "weather, statistical datasets.",
    "reference": "Durable factual material with no publication moment: an "
                 "encyclopedia entry, a dictionary or glossary definition, an "
                 "academic paper, a legal text, a recipe, medical information, a "
                 "patent. A Wikipedia entry is reference. A recipe is reference.",
    "discussion": "Content produced by a back-and-forth between people: questions "
                  "and answers, forum threads, comment threads, social posts.",
    "code": "A software artefact page: a repository, a published package, a code "
            "snippet, a pull request.",
    "media": "A page whose primary object is audio or video: a talk, a podcast "
             "episode, an image gallery, an album.",
    "profile": "A page describing an entity: a person, an organisation, a place.",
    "event": "A page for something happening at a time and place: a conference, "
             "meetup, webinar, workshop.",
}


def api_key() -> str:
    k = os.environ.get("JEV_API_KEY")
    if k:
        return k.strip()
    for d in (pathlib.Path.cwd(), *pathlib.Path.cwd().parents,
              pathlib.Path(r"C:\Users\sync\OneDrive\Documents\GitHub\samsara-net")):
        f = d / ".env"
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.startswith("JEV_API_KEY") and "=" in line:
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("JEV_API_KEY not found in the environment or a .env file")


def questions():
    """The parent choice plus one speculative subtype choice per category."""
    qs = {"parent": {
        "type": "choice",
        "instructions": {
            "task": "Classify this web page into one SDF content category.",
            "guidance": "Judge by what the page IS, not what it is about. A paper, "
                        "encyclopedia entry or glossary definition is reference, "
                        "not article. Vendor docs for a product are documentation, "
                        "not reference. A question with answers under it is "
                        "discussion, even on a technical site.",
        },
        "criteria": dict(CRITERIA),
    }}
    for p, subs in SUBTYPES.items():
        qs[f"sub_{p}"] = {
            "type": "choice",
            "instructions": {
                "premise": f"Assume this page belongs to the '{p}' category.",
                "task": f"Under that assumption, which specific kind of {p} is it?",
            },
            "criteria": {s: None for s in subs},
        }
    return qs


def state_for(url, title, text, limit=6000):
    return {"url": url, "title": title, "page_text": (text or "")[:limit]}


def main():
    import httpx

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    which = sys.argv[2] if len(sys.argv) > 2 else "test"
    rows, total = split(n, which)
    key = api_key()
    qs = questions()
    print(f"  {total} pages carry a URL-derived gold label; asking Jev about "
          f"{len(rows)} from the {which} half")
    print(f"  {len(qs)} questions per page in one request "
          f"(1 parent + {len(SUBTYPES)} speculative subtypes)\n")

    okp, oks, sp, ss, cp_, times = [], [], [], [], [], []
    tok_in = tok_out = 0
    conf_mat = Counter()
    model_seen = ""
    with httpx.Client(timeout=90) as client:
        for i, (url, title, spar, ssub, text, (gp, gs)) in enumerate(rows):
            body = {"state": state_for(url, title, text), "model": MODEL,
                    "questions": qs}
            t0 = time.time()
            try:
                r = client.post(ENDPOINT, json=body, headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json"})
                r.raise_for_status()
                d = r.json()
            except Exception as e:                        # noqa: BLE001
                print(f"  request {i} failed: {type(e).__name__}: {str(e)[:90]}")
                continue
            times.append(time.time() - t0)
            model_seen = d.get("model", model_seen)
            u = d.get("usage") or {}
            tok_in += u.get("input_tokens", 0)
            tok_out += u.get("output_tokens", 0)

            a = d["answers"]
            parent = a["parent"]["choice"]
            sub = a[f"sub_{parent}"]["choice"]
            okp.append(parent == gp)
            cp_.append(a["parent"].get("confidence", float("nan")))
            sp.append(spar == gp)
            if gs:
                oks.append(parent == gp and sub == gs)
                ss.append(spar == gp and ssub == gs)
            if parent != gp:
                conf_mat[(gp, parent)] += 1

    if not okp:
        print("  no successful requests")
        return
    okp, sp = np.array(okp, float), np.array(sp, float)
    oks, ss = np.array(oks, float), np.array(ss, float)
    cp_ = np.array(cp_, float)
    maj = Counter(r[5][0] for r in rows).most_common(1)[0]

    print(f"  model {model_seen}, {len(okp)} pages answered\n")
    print(f"  {'':34s} {'Jev':>8s} {'local 27B':>11s} {'stored (GPT)':>13s}")
    print(f"  parent_type  (n={len(okp):3d})              {okp.mean():8.3f} "
          f"{0.928:11.3f} {sp.mean():13.3f}")
    print(f"  full path    (n={len(oks):3d})              {oks.mean():8.3f} "
          f"{0.912:11.3f} {ss.mean():13.3f}")
    print(f"  always say {maj[0]!r:<14s}          {maj[1]/len(rows):8.3f}")
    print(f"\n  median {np.median(times):.2f} s per page for {len(qs)} questions in "
          f"one request")
    print(f"  {tok_in} input tokens, {tok_out} output tokens over {len(okp)} pages "
          f"({tok_in//max(1,len(okp))} in per page)")

    print("\n  parent calibration:")
    for lo, hi in ((0, .6), (.6, .85), (.85, .97), (.97, 1.01)):
        m = (cp_ >= lo) & (cp_ < hi)
        if m.sum():
            print(f"    conf {lo:.2f}-{hi:.2f}: n={int(m.sum()):3d}  "
                  f"mean {cp_[m].mean():.2f}  correct {okp[m].mean():.2f}")

    if conf_mat:
        print("\n  parent errors (gold -> Jev):")
        for (g, p), c in conf_mat.most_common(8):
            print(f"    {g:16s} -> {p:16s} {c}")
    print("\n  The local column is probe_sdf.py on the same held-out half, and the")
    print("  stored column is the SDF pipeline's own labels, both on this gold.")


if __name__ == "__main__":
    main()
