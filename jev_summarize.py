"""Summarise a page with Jev by SELECTING its sentences, not writing new ones.

Jev does not generate text. It returns typed judgments, so it cannot write a
summary. It can choose one: split the page into sentences in code, ask Jev which
carry information a reader would need, and keep the best within a token budget.

That is not a workaround. Two measurements from this repository say it is the
better artefact for SDF's purpose:

  probe_summary.py found that an abstractive SDF document answers only 0.500 of
  the questions that actually need the page, while the same number of tokens of
  raw page text answers 0.900. A brief and five key points capture what a page is
  about and discard what it says.

  A selected sentence is copied, so it cannot be hallucinated, and the citation
  is free: every sentence is still findable in the source.

The whole page goes in `state` once and one question per sentence goes in the
same request, which is TypeSafe's fan-out. Sentences are referenced by path so
the text is not repeated per question.

    python jev_summarize.py <url> [--budget 600]
"""
from __future__ import annotations

import json
import re
import sys
import time

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
SENT = re.compile(r"(?<=[.!?])\s+")

QUESTION = (
    "Does sentence {i} in `sentences` carry information someone would need in "
    "order to act on this page without reading it: a specific figure, name, "
    "date, condition, limit, mechanism, decision or outcome? Answer no for "
    "navigation, boilerplate, promotion, or framing that states nothing "
    "checkable."
)


def api_key():
    from probe_jev import api_key as k
    return k()


def sentences(text, lo=45, hi=340, cap=120):
    out = []
    for para in (text or "").split("\n"):
        for s in SENT.split(para):
            s = " ".join(s.split())
            if lo <= len(s) <= hi and " " in s:
                out.append(s)
            if len(out) >= cap:
                return out
    return out


def score_sentences(client, key, url, title, sents, batch=40):
    """One request per batch of sentences, the page sent once in each."""
    scores, usage = [], {"input_tokens": 0, "output_tokens": 0}
    for off in range(0, len(sents), batch):
        chunk = sents[off:off + batch]
        state = {"url": url, "title": title, "sentences": chunk}
        qs = {f"s{j}": {"type": "noul", "instructions": QUESTION.format(i=j)}
              for j in range(len(chunk))}
        r = client.post(ENDPOINT, json={"state": state, "model": MODEL,
                                        "questions": qs},
                        headers={"Authorization": f"Bearer {key}"})
        r.raise_for_status()
        d = r.json()
        for j in range(len(chunk)):
            scores.append(float(d["answers"][f"s{j}"]["noul"]))
        u = d.get("usage") or {}
        usage["input_tokens"] += u.get("input_tokens", 0)
        usage["output_tokens"] += u.get("output_tokens", 0)
    return scores, usage


def pick(sents, scores, budget_tokens, approx=4.0):
    """Keep the highest-scoring sentences that fit, then restore reading order."""
    order = {s: i for i, s in enumerate(sents)}
    ranked = sorted(zip(scores, sents), key=lambda x: -x[0])
    kept, used = [], 0
    for sc, s in ranked:
        cost = len(s) / approx
        if used + cost > budget_tokens:
            continue
        kept.append((sc, s))
        used += cost
        if used >= budget_tokens - 12:
            break
    kept.sort(key=lambda x: order.get(x[1], 0))
    return [s for _, s in kept], [sc for sc, _ in kept]


def summarize(client, key, url, title, text, budget_tokens=600):
    sents = sentences(text)
    if not sents:
        return {"sentences": [], "scores": [], "usage": {}, "n_candidates": 0}
    t0 = time.time()
    scores, usage = score_sentences(client, key, url, title, sents)
    kept, kept_scores = pick(sents, scores, budget_tokens)
    return {"sentences": kept, "scores": kept_scores, "usage": usage,
            "n_candidates": len(sents), "seconds": time.time() - t0}


def main():
    import httpx
    import trafilatura
    from sdf_convert import UA, allowed, api_url

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    budget = 600
    if "--budget" in sys.argv:
        budget = int(sys.argv[sys.argv.index("--budget") + 1])
    url = args[0]

    if not allowed(url):
        raise SystemExit("robots.txt disallows fetching this path")
    r = httpx.get(api_url(url), headers={"User-Agent": UA}, timeout=30,
                  follow_redirects=True)
    r.raise_for_status()
    text = trafilatura.extract(r.text, include_comments=False,
                               include_tables=True) or ""
    m = re.search(r"<title[^>]*>(.*?)</title>", r.text, re.S | re.I)
    title = " ".join(m.group(1).split()) if m else url

    key = api_key()
    with httpx.Client(timeout=90) as client:
        out = summarize(client, key, url, title, text, budget)

    print(f"  {url}")
    print(f"  page {len(r.text)//1024} KB, {len(text)} chars of text, "
          f"{out['n_candidates']} candidate sentences")
    print(f"  kept {len(out['sentences'])} in {out.get('seconds', 0):.2f} s, "
          f"{out['usage'].get('input_tokens', 0)} input tokens\n")
    for s, sc in zip(out["sentences"], out["scores"]):
        print(f"  [{sc:.2f}] {s}")


if __name__ == "__main__":
    main()
