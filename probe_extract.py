"""Can a typed reader extract values, not just choose categories?

Classification is the easy case: the options are given. Extraction has to produce
a headline, an author, a date, which are not a fixed list. The move is to notice
that the answer is always a span already printed on the page, harvest those spans
with regular expressions, and let the model pick one (sdf_spans.py).

Two things are then true by construction: the value is a literal substring of the
source, so it cannot be hallucinated, and the choice is closed, so it reads from
the logits like any other field. "none" is always available, so a page that does
not state an author can say so.

Measured against what the publisher declares about the page in JSON-LD and meta
tags (sdf_ld.py). The model never sees that markup; it sees the rendered text
with the tags stripped. So the gold was written by neither side.

The comparison is generation: the same model, the same page, asked to emit the
same fields as JSON. That is what the SDF worker does today.

    python probe_extract.py <model.gguf> [n]
"""
from __future__ import annotations

import json
import re
import sys
import time

import numpy as np

from yantrik_inference import Field, open_model
import sdf_spans as spans
from sdf_ld import from_html, same
from sdf_convert import UA, allowed, api_url

FIELDS = ("headline", "author", "publish_date")

FEEDS = ("https://blog.cloudflare.com/rss/",
         "https://techcrunch.com/feed/",
         "https://arstechnica.com/feed/",
         "https://www.theguardian.com/world/rss",
         "https://www.theguardian.com/science/rss")

QUESTION = {
    "headline": "Which lettered headline candidate is this page's actual title?",
    "author": "Which lettered author candidate is the person who wrote this page?",
    "publish_date": "Which lettered date candidate is when this page was published?",
}


def feed_urls(limit_per_feed=14):
    import httpx
    out = []
    for f in FEEDS:
        try:
            r = httpx.get(f, headers={"User-Agent": UA}, timeout=25,
                          follow_redirects=True)
            links = re.findall(r"<link>\s*(https?://[^<\s]+?)\s*</link>", r.text)
            links += re.findall(r'<link[^>]+href="(https?://[^"]+)"[^>]*/?>', r.text)
            good = [u for u in links if not u.rstrip("/").endswith(("rss", "feed"))]
            out += good[:limit_per_feed]
        except Exception:                                 # noqa: BLE001
            continue
    return list(dict.fromkeys(out))


def page(url):
    """The page as a reader sees it, plus what the publisher declares about it.

    The text is the VISIBLE page, not the article body. trafilatura in precision
    mode returns the article, which on every live page tested had the headline,
    byline and dateline stripped out: the candidate lists could not hold the
    answer because the text did not. The gold still comes from <head> and the
    JSON-LD block, neither of which survives into the visible text.
    """
    import httpx
    if not allowed(url):
        return None
    r = httpx.get(api_url(url), headers={"User-Agent": UA}, timeout=25,
                  follow_redirects=True)
    if r.status_code != 200:
        return None
    text = spans.visible_text(r.text)
    if len(text) < 800:
        return None
    g = from_html(r.text)
    if not any(k in g for k in FIELDS):
        return None
    return {"url": url, "text": text, "gold": g}


def build_record(reader, p, cands, budget):
    head = (f"PAGE\nURL: {p['url']}\n\n{spans.render(cands)}\n\nPage text:\n")
    fixed = len(reader.tok(reader.head + reader.framed(head), bos=True))
    ids = reader.tok(p["text"], special=False)[:max(0, budget - fixed)]
    return head + reader.llm.detokenize(ids).decode("utf-8", "replace")


def gen_extract(chat, p, budget_chars=7000):
    """The baseline: ask for the same fields as JSON and parse the reply."""
    ask = (f"URL: {p['url']}\n\nPage text:\n{p['text'][:budget_chars]}\n\n"
           'Extract these fields as JSON: {"headline": ..., "author": ..., '
           '"publish_date": ...}. Use null for anything the page does not state. '
           "Reply with JSON only.")
    out = "".join(chat.stream([dict(role="user", content=ask)],
                              max_tokens=220, temperature=0.0))
    m = re.search(r"\{.*\}", out, re.S)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    return {k: (str(v).strip() if v not in (None, "", "null") else None)
            for k, v in d.items() if k in FIELDS}


def main():
    gguf = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    urls = feed_urls()
    print(f"  {len(urls)} article URLs from public feeds; fetching until {n} usable\n")

    pages = []
    for u in urls:
        try:
            p = page(u)
        except Exception:                                 # noqa: BLE001
            p = None
        if p:
            pages.append(p)
        if len(pages) >= n:
            break
    print(f"  {len(pages)} pages carry publisher-declared values to score against\n")

    reader, chat = open_model(gguf, decide_ctx=65536, decide_seq=16, chat_ctx=16384)
    budget = reader.per_seq - 80

    sel = {f: [] for f in FIELDS}          # correct?
    gen = {f: [] for f in FIELDS}
    onpage = {f: [] for f in FIELDS}       # is the produced value on the page?
    conf = {f: [] for f in FIELDS}
    cover = {f: [] for f in FIELDS}        # was the right answer among candidates?
    t_sel = t_gen = 0.0

    for p in pages:
        cands = spans.menu(p["url"], p["text"], FIELDS)
        rec = build_record(reader, p, cands, budget)
        fs = [Field(QUESTION[f], spans.options(cands[f])) for f in FIELDS]
        t0 = time.time()
        got = reader.read(rec, fs)
        t_sel += time.time() - t0

        t0 = time.time()
        g = gen_extract(chat, p)
        t_gen += time.time() - t0

        low = p["text"].lower()
        for f, a in zip(FIELDS, got):
            want = p["gold"].get(f)
            if want is None:
                continue
            picked = spans.resolve(a.answer, cands[f])
            sel[f].append(bool(picked and same(picked, want, f)))
            conf[f].append(a.confidence)
            cover[f].append(any(same(c, want, f) for c in cands[f]))
            v = g.get(f)
            gen[f].append(bool(v and same(v, want, f)))
            if v:
                onpage[f].append(re.sub(r"\s+", " ", v.lower())[:60] in
                                 re.sub(r"\s+", " ", low))

    print(f"  {'field':14s} {'n':>3s} {'select':>7s} {'generate':>9s} "
          f"{'candidate had it':>17s} {'gen value on page':>18s}")
    for f in FIELDS:
        if not sel[f]:
            continue
        s, gg = np.array(sel[f], float), np.array(gen[f], float)
        cv = np.array(cover[f], float)
        op = np.array(onpage[f], float) if onpage[f] else np.array([np.nan])
        print(f"  {f:14s} {len(s):3d} {s.mean():7.3f} {gg.mean():9.3f} "
              f"{cv.mean():17.3f} {np.nanmean(op):18.3f}")

    allsel = np.array([x for f in FIELDS for x in sel[f]], float)
    allgen = np.array([x for f in FIELDS for x in gen[f]], float)
    allop = np.array([x for f in FIELDS for x in onpage[f]], float)
    print(f"\n  overall  select {allsel.mean():.3f}   generate {allgen.mean():.3f}"
          f"   over {len(allsel)} field values")
    print(f"  a selected value is a literal span of the page 100% of the time;")
    print(f"  a generated one was {allop.mean()*100:.0f}% of the time")
    print(f"\n  select {t_sel/len(pages):.2f} s per page for {len(FIELDS)} fields, "
          f"generate {t_gen/len(pages):.2f} s")

    print("\n  The harvester is the ceiling: where the candidate list does not hold")
    print("  the answer, selection cannot be right however good the model is.")
    print(f"  {'field':14s} {'harvester found it':>19s} {'model picked it there':>22s}")
    for f in FIELDS:
        if cover[f]:
            cv = np.array(cover[f], float)
            sv = np.array(sel[f], float)
            hit = cv > 0
            if hit.any():
                print(f"  {f:14s} {cv.mean():19.3f} {sv[hit].mean():22.3f}")


if __name__ == "__main__":
    main()
