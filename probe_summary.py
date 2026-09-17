"""Does an SDF document substitute for the page it came from?

That is what SDF is for: convert once so nobody has to read the page again. The
test that matters is therefore not whether a headline matches a meta tag. It is
whether an agent asked something about the page gets the same answer from the
2 KB document as it would from the 300 KB page.

So: take a page, write questions about it, and answer them four ways.

    full page       the reference. Not truth, just what the page supports.
    SDF document    the artefact under test.
    same-size page  the control that matters. The SDF document costs N tokens,
                    so this is the first N tokens of the raw page text. If plain
                    truncation does as well, the conversion earns nothing.
    url and title   the floor. Questions answerable from here needed no page at
                    all and say nothing about either artefact.

The questions are generated from the full text, but their answer key is never
used: the reference is what the model answers holding the whole page. This
measures preservation, which is the property SDF actually claims.

Each condition is one batched read over its own prefill, so six questions cost
one pass per condition.

    python probe_summary.py <model.gguf> [n]
"""
from __future__ import annotations

import json
import re
import sys
import time

import numpy as np

from yantrik_inference import Field, open_model
from probe_extract import feed_urls
from sdf_convert import UA, allowed, api_url, typed_fields, ENTITY_TYPES, BCP47
from probe_sdf import PARENTS, record_for, type_path

LETTERS = ("A", "B", "C", "D")


def fetch_text(url):
    import httpx
    import trafilatura
    if not allowed(url):
        return None
    r = httpx.get(api_url(url), headers={"User-Agent": UA}, timeout=25,
                  follow_redirects=True)
    if r.status_code != 200:
        return None
    text = trafilatura.extract(r.text, include_comments=False,
                               include_tables=True) or ""
    if len(text) < 2500:
        return None
    m = re.search(r"<title[^>]*>(.*?)</title>", r.text, re.S | re.I)
    title = re.sub(r"\s+", " ", m.group(1)).strip() if m else url
    return {"url": url, "text": text, "title": title,
            "html_bytes": len(r.text.encode())}


def gen(chat, prompt, max_tokens=420):
    return "".join(chat.stream([dict(role="user", content=prompt)],
                               max_tokens=max_tokens, temperature=0.0))


def as_json(s):
    m = re.search(r"\{.*\}|\[.*\]", s or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def make_sdf(reader, chat, p, budget):
    """The artefact: typed fields read, open fields generated."""
    rec = record_for(reader, p["url"], p["title"], p["text"], budget)
    a = reader.read(rec, typed_fields())
    parent, sub, cp, cs = type_path(a)
    body = p["text"][:9000]
    s = as_json(gen(chat, f"{body}\n\nSummarise this page as "
                          '{"brief": "one sentence", "detailed": "three sentences", '
                          '"key_points": ["...", "...", "...", "...", "..."]}. '
                          "Reply with JSON only."))
    names = as_json(gen(chat, f"{body}\n\nList up to eight named entities on this "
                              'page as ["name", ...]. Reply with JSON only.', 200))
    names = [str(x)[:60] for x in names][:8] if isinstance(names, list) else []
    ents = []
    if names:
        got = reader.read(rec, [Field(f"On this page, what kind of thing is {n!r}?",
                                      ENTITY_TYPES) for n in names])
        ents = [{"name": n, "type": g.answer} for n, g in zip(names, got)]
    topics = as_json(gen(chat, f"{body}\n\nList five topic tags for this page as "
                               '["...", ...]. Reply with JSON only.', 120))
    doc = {
        "sdf_version": "0.2.0", "canonical_url": p["url"],
        "parent_type": parent, "type": sub,
        "language": BCP47[a[11].answer],
        "source": {"url": p["url"], "title": p["title"]},
        "summary": s if isinstance(s, dict) else {},
        "entities": ents,
        "topics": [str(t)[:40] for t in topics][:5] if isinstance(topics, list) else [],
        "metadata": {"paywalled": a[12].answer == "yes",
                     "time_sensitive": a[13].answer == "yes",
                     "has_named_author": a[14].answer == "yes",
                     "commercial_intent": a[15].answer == "yes"},
    }
    return doc


def make_questions(chat, p, k=6):
    """Questions about this page, four options each. The answer key is unused."""
    out = gen(chat, f"{p['text'][:9000]}\n\nWrite {k} multiple-choice questions "
                    "about specific facts stated in this page. Each must have four "
                    "plausible options, only one supported by the page. Do not ask "
                    "about the title or the author. Reply with JSON only: "
                    '[{"q": "...", "options": ["...", "...", "...", "..."]}]', 900)
    qs = as_json(out)
    if not isinstance(qs, list):
        return []
    good = []
    for item in qs:
        if not isinstance(item, dict):
            continue
        o = item.get("options")
        q = item.get("q")
        if isinstance(q, str) and isinstance(o, list) and len(o) == 4:
            good.append({"q": q.strip()[:220],
                         "options": [str(x).strip()[:90] for x in o]})
    return good[:k]


def ask(reader, context, qs, budget, reader_tok):
    """One batched read: every question against the same context prefill."""
    lines = []
    for i, item in enumerate(qs):
        lines.append(f"Question {i + 1}: {item['q']}")
        for L, o in zip(LETTERS, item["options"]):
            lines.append(f"  {L}) {o}")
    head = f"{context}\n\n" + "\n".join(lines) + "\n"
    fixed = len(reader_tok(head))
    if fixed > budget:                      # trim the context, never the questions
        room = budget - (fixed - len(reader_tok(context)))
        ids = reader.tok(context, special=False)[:max(0, room)]
        context = reader.llm.detokenize(ids).decode("utf-8", "replace")
        head = f"{context}\n\n" + "\n".join(lines) + "\n"
    fs = [Field(f"For question {i + 1} above, which option does the text support?",
                LETTERS) for i in range(len(qs))]
    return reader.read(head, fs), len(reader_tok(head))


def main():
    gguf = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    reader, chat = open_model(gguf, decide_ctx=65536, decide_seq=16, chat_ctx=16384)
    budget = reader.per_seq - 120
    tok = lambda s: reader.tok(reader.head + reader.framed(s), bos=True)

    urls = feed_urls()
    print(f"  {len(urls)} candidate URLs; using the first {n} that fetch\n")

    rows, sizes = [], []
    for u in urls:
        if len(rows) >= n:
            break
        try:
            p = fetch_text(u)
        except Exception:                                 # noqa: BLE001
            p = None
        if not p:
            continue
        t0 = time.time()
        doc = make_sdf(reader, chat, p, budget)
        qs = make_questions(chat, p)
        if len(qs) < 4:
            continue
        build_s = time.time() - t0

        doc_text = json.dumps(doc, ensure_ascii=False, indent=1)
        n_doc = len(reader.tok(doc_text, special=False))
        trunc = reader.llm.detokenize(
            reader.tok(p["text"], special=False)[:n_doc]).decode("utf-8", "replace")
        floor = f"URL: {p['url']}\nTitle: {p['title']}"

        conds = {"full page": p["text"], "SDF document": doc_text,
                 "same-size page": trunc, "url+title": floor}
        ans, toks = {}, {}
        for name, ctx in conds.items():
            a, t = ask(reader, ctx, qs, budget, tok)
            ans[name] = [x.answer for x in a]
            toks[name] = t
        ref = ans["full page"]
        rows.append({k: np.mean([x == y for x, y in zip(v, ref)])
                     for k, v in ans.items()} | {"nq": len(qs)})
        sizes.append(toks | {"html_kb": p["html_bytes"] / 1024, "build_s": build_s})
        print(f"  {p['url'][8:66]:66s} {len(qs)}q  "
              + "  ".join(f"{k}={rows[-1][k]:.2f}" for k in
                          ("SDF document", "same-size page", "url+title")))

    if not rows:
        print("  no usable pages")
        return
    nq = sum(r["nq"] for r in rows)
    print(f"\n  {len(rows)} pages, {nq} questions, agreement with the full-page answer\n")
    print(f"  {'context given':18s} {'tokens':>8s} {'agreement':>10s}")
    for k in ("full page", "SDF document", "same-size page", "url+title"):
        t = np.mean([s[k] for s in sizes])
        print(f"  {k:18s} {t:8.0f} {np.mean([r[k] for r in rows]):10.3f}")
    print(f"\n  page HTML averages {np.mean([s['html_kb'] for s in sizes]):.0f} KB; "
          f"the document is {np.mean([s['SDF document'] for s in sizes]):.0f} tokens")
    print(f"  building one document took {np.mean([s['build_s'] for s in sizes]):.1f} s")
    print("\n  The control is 'same-size page'. The document only earns its keep")
    print("  where it beats plain truncation to the same token count.")


if __name__ == "__main__":
    main()
