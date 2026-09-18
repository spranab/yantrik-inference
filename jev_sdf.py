"""A whole SDF document from Jev: classify, select, type. No generated text.

The flow SDF's worker runs today is generate-and-parse: ask a model for JSON,
parse it, repair the JSON when it is malformed, escalate to a bigger model when
confidence is low. Every step of this one is a typed judgment instead.

    1. classify   one Choice for the parent category plus one speculative Choice
                  per category for its subtype, all in a single request. Code
                  takes the subtype belonging to the chosen parent, so the type
                  path cannot be inconsistent.
    2. select     one Noul per sentence: does this carry something a reader would
                  need? The page goes in `state` once and the sentences are
                  referenced by path. Kept sentences are COPIED, so they cannot
                  be hallucinated and each stays findable in the source.
    3. type       one Choice per kept sentence over SDF's own claim enum
                  (fact, opinion, prediction, instruction, definition).

The result populates `claims`, a field the published schema declares and the
shipping pipeline leaves empty, and it is validated against that schema before
it is written. Measured justification for preferring this over a written summary
is in probe_jev_summary.py: on questions that actually need the page, a written
summary answered 0.500 while raw text of the same size answered 0.900, so the
summary was losing what agents ask about.

    python jev_sdf.py <url> [url ...] [--budget 500] [--out DIR]
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from probe_jev import ENDPOINT, MODEL, CRITERIA, api_key, questions as type_questions
from probe_sdf import PARENTS, SUBTYPES
from jev_summarize import summarize
from sdf_convert import UA, allowed, api_url, SCHEMA, validator

CLAIM_TYPES = ("fact", "opinion", "prediction", "instruction", "definition")


def fetch(url):
    import httpx
    import trafilatura
    if not allowed(url):
        raise PermissionError("robots.txt disallows fetching this path")
    r = httpx.get(api_url(url), headers={"User-Agent": UA}, timeout=30,
                  follow_redirects=True)
    r.raise_for_status()
    text = trafilatura.extract(r.text, include_comments=False,
                               include_tables=True) or ""
    meta = trafilatura.extract_metadata(r.text)
    title = (getattr(meta, "title", None) or "").strip()
    if not title:
        m = re.search(r"<title[^>]*>(.*?)</title>", r.text, re.S | re.I)
        title = " ".join(m.group(1).split()) if m else url
    return {"html_bytes": len(r.text.encode()), "text": text, "title": title,
            "author": getattr(meta, "author", None),
            "site_name": getattr(meta, "sitename", None),
            "published_at": getattr(meta, "date", None)}


def post(client, key, state, qs):
    r = client.post(ENDPOINT, json={"state": state, "model": MODEL,
                                    "questions": qs},
                    headers={"Authorization": f"Bearer {key}"})
    r.raise_for_status()
    return r.json()


def classify(client, key, url, title, text):
    d = post(client, key, {"url": url, "title": title,
                           "page_text": (text or "")[:6000]}, type_questions())
    a = d["answers"]
    parent = a["parent"]["choice"]
    sub = a[f"sub_{parent}"]
    return (parent, sub["choice"], a["parent"].get("confidence", 0.0),
            sub.get("confidence", 0.0), (d.get("usage") or {}))


def type_claims(client, key, url, title, sents, batch=25):
    """SDF fixes claim type to five values, so read it rather than guess it."""
    out = []
    for off in range(0, len(sents), batch):
        chunk = sents[off:off + batch]
        qs = {f"c{j}": {
            "type": "choice",
            "instructions": {
                "task": f"Classify statement {j} in `statements`.",
                "guidance": "Judge the statement itself, not the page around it.",
            },
            "criteria": {
                "fact": "A checkable assertion about how things are or were.",
                "opinion": "A judgement, preference or evaluation.",
                "prediction": "A claim about the future or an expected outcome.",
                "instruction": "Tells the reader to do something, or how to.",
                "definition": "States what a term or thing means.",
            }} for j in range(len(chunk))}
        d = post(client, key, {"url": url, "title": title, "statements": chunk}, qs)
        for j in range(len(chunk)):
            out.append(d["answers"][f"c{j}"]["choice"])
    return out


def build(client, key, url, budget=500):
    t0 = time.time()
    page = fetch(url)
    t_fetch = time.time() - t0

    t1 = time.time()
    parent, sub, cp, cs, _ = classify(client, key, url, page["title"], page["text"])
    sel = summarize(client, key, url, page["title"], page["text"], budget)
    kinds = type_claims(client, key, url, page["title"], sel["sentences"]) \
        if sel["sentences"] else []
    t_jev = time.time() - t1

    digest = hashlib.sha256((page["text"] or url).encode()).hexdigest()
    claims = [{"statement": s, "type": k, "confidence": round(float(sc), 3)}
              for s, k, sc in zip(sel["sentences"], kinds, sel["scores"])]
    brief = sel["sentences"][0][:300] if sel["sentences"] else page["title"][:300]

    doc = {
        "sdf_version": "0.2.0",
        "id": f"sdf:{digest}",
        "sdf": {"version": "0.2", "dialect": "core", "features": []},
        "canonical_url": url,
        "parent_type": parent,
        "type": sub,
        "type_data": {},
        "source": {k: v for k, v in {
            "url": url, "title": page["title"], "author": page["author"],
            "site_name": page["site_name"] or urlparse(url).hostname,
            "published_at": page["published_at"],
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }.items() if v not in ("", None)},
        # brief is the page's own highest-scoring sentence, not a written one
        "summary": {"brief": brief,
                    "key_points": [c["statement"] for c in claims][:10]},
        "claims": claims,
        "provenance": {
            "converter": "yantrik-inference jev_sdf",
            "model_used": "jev (typed judgments only, no generated text)",
            "conversion_confidence": round(min(cp, cs), 4),
            "content_hash": f"sha256:{digest}",
        },
    }
    doc = {k: v for k, v in doc.items() if v not in ([], None, "")}
    stats = {"fetch_s": t_fetch, "jev_s": t_jev, "html_kb": page["html_bytes"] / 1024,
             "sdf_kb": len(json.dumps(doc).encode()) / 1024,
             "candidates": sel["n_candidates"], "kept": len(claims),
             "cp": cp, "cs": cs,
             "tokens": sel["usage"].get("input_tokens", 0)}
    return doc, stats


def main():
    import httpx

    argv, opts, pos = sys.argv[1:], {}, []
    i = 0
    while i < len(argv):
        if argv[i].startswith("--"):
            opts[argv[i][2:]] = argv[i + 1] if i + 1 < len(argv) else ""
            i += 2
        else:
            pos.append(argv[i]); i += 1
    budget = int(opts.get("budget", 500))
    out = Path(opts["out"]) if "out" in opts else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    key = api_key()
    v = validator()
    print(f"  classify, select and type entirely with Jev; budget {budget} tokens; "
          f"schema validation {'on' if v else 'UNAVAILABLE'}\n")
    ok = bad = 0
    with httpx.Client(timeout=120) as client:
        for url in pos:
            try:
                doc, st = build(client, key, url, budget)
            except Exception as e:                        # noqa: BLE001
                print(f"  FAILED {url[:70]}\n         {type(e).__name__}: {str(e)[:90]}")
                bad += 1
                continue
            errs = sorted(v.iter_errors(doc), key=lambda e: list(e.path)) if v else []
            ok += not errs
            bad += bool(errs)
            print(f"  {doc['parent_type']}.{doc['type']:<15s} "
                  f"{st['cp']*100:3.0f}%/{st['cs']*100:3.0f}%  "
                  f"{st['html_kb']:6.0f}KB -> {st['sdf_kb']:5.1f}KB  "
                  f"{st['kept']:2d}/{st['candidates']:3d} sentences kept  "
                  f"jev {st['jev_s']:.2f}s  {'VALID' if not errs else 'INVALID'}")
            print(f"      {url[:100]}")
            for c in doc.get("claims", [])[:3]:
                print(f"      [{c['type']:11s} {c['confidence']:.2f}] "
                      f"{c['statement'][:82]}")
            for e in errs[:3]:
                print(f"      SCHEMA ERROR at {list(e.path)}: {e.message[:80]}")
            if out:
                name = re.sub(r"[^a-zA-Z0-9]+", "-", url)[:90] + ".json"
                (out / name).write_text(json.dumps(doc, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    print(f"\n  {ok} schema-valid, {bad} not")
    if out:
        print(f"  written to {out}")


if __name__ == "__main__":
    main()
