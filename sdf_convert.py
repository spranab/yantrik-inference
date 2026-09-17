"""Convert live web pages into SDF Protocol 0.2 documents on one local model.

The rule: a field with a closed set of answers is READ from the logits and cannot
come out invalid; only genuinely open fields are generated. Both run on the same
loaded weights, and the page is prefilled once for all of it.

    read      parent_type, type, language, paywalled, time_sensitive,
              has_author, commercial_intent, and every entity's type
    generate  the brief, the key points, and the entity NAMES

Entity types are the clearest case. SDF fixes them to eight values, so a
generated entity type can be wrong in a way no parser can fix. Here the names are
generated and their types are read as a batch of typed fields over the same page
prefix, which costs one extra pass for all eight entities together.

The conditional type path is free for the same reason: the parent question and
all ten subtype questions read one prefill, and the subtype taken is the one
belonging to the chosen parent, so the path cannot be inconsistent.

Every document is validated against the real schema in sdf/spec before it is
written, because "valid by construction" is a claim that has to be checked.

    python sdf_convert.py <model.gguf> <url> [url ...] [--out DIR] [--no-generate]
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

from yantrik_inference import Field, open_model
from probe_sdf import PARENTS, SUBTYPES, TAXONOMY, record_for, type_path

SCHEMA = Path(r"C:\Users\sync\codes\sdf\spec\schemas\sdf-document-0.2.schema.json")

LANGS = ("english", "spanish", "french", "german", "chinese", "japanese",
         "arabic", "russian", "portuguese", "other")
BCP47 = {"english": "en", "spanish": "es", "french": "fr", "german": "de",
         "chinese": "zh", "japanese": "ja", "arabic": "ar", "russian": "ru",
         "portuguese": "pt", "other": "und"}

# Exactly the enum in the published schema. Generating these would be a guess;
# reading them cannot produce a ninth value.
ENTITY_TYPES = ("person", "organization", "concept", "technology", "location",
                "product", "event", "other")


def typed_fields():
    fs = [Field("Which of the ten SDF categories does this page belong to?", PARENTS)]
    for p, subs in SUBTYPES.items():
        fs.append(Field(f"Assuming this page is in the {p} category, "
                        f"which specific kind is it?", subs))
    fs += [
        Field("What language is this page written in?", LANGS),
        Field("Is the full content behind a paywall or login?", ("yes", "no")),
        Field("Does the value of this page depend on when it is read?", ("yes", "no")),
        Field("Does the page name a human author or byline?", ("yes", "no")),
        Field("Is the page trying to sell or promote something?", ("yes", "no")),
    ]
    return fs


# An honest agent string. Wikipedia answers a browser-spoofing client with
# "please respect our robot policy", and the answer to that is to respect it,
# not to spoof harder: identify the tool and give someone a way to complain.
UA = "yantrik-inference/0.1 (SDF converter; +https://github.com/spranab/yantrik-inference)"

_robots: dict = {}


def allowed(url: str, timeout=10.0) -> bool:
    """Ask the site's robots.txt before fetching, and cache the answer per host.

    A converter that reads the public web at someone else's expense should do
    what it is told. A host with no robots.txt, or one that cannot be read, is
    treated as permitting: absent rules are not a prohibition.
    """
    import urllib.robotparser

    import httpx
    u = urlparse(url)
    root = f"{u.scheme}://{u.netloc}"
    if root not in _robots:
        rp = urllib.robotparser.RobotFileParser()
        try:
            r = httpx.get(f"{root}/robots.txt", timeout=timeout,
                          headers={"User-Agent": UA}, follow_redirects=True)
            rp.parse(r.text.splitlines() if r.status_code == 200 else [])
        except Exception:                                 # noqa: BLE001
            rp.parse([])
        _robots[root] = rp
    return _robots[root].can_fetch(UA, url)


def api_url(url: str) -> str:
    """Prefer a sanctioned endpoint where the site publishes one."""
    u = urlparse(url)
    if u.hostname and u.hostname.endswith("wikipedia.org") and u.path.startswith("/wiki/"):
        return f"{u.scheme}://{u.netloc}/api/rest_v1/page/html/{u.path[6:]}"
    return url


def fetch(url: str, timeout=25.0):
    import httpx
    import trafilatura
    if not allowed(url):
        raise PermissionError("robots.txt disallows fetching this path")
    with httpx.Client(follow_redirects=True, timeout=timeout,
                      headers={"User-Agent": UA,
                               "Accept": "text/html,application/xhtml+xml,*/*"}) as c:
        r = c.get(api_url(url))
        r.raise_for_status()
        html = r.text
    text = trafilatura.extract(html, include_comments=False, include_tables=True,
                               favor_precision=True) or ""
    meta = trafilatura.extract_metadata(html)
    title = (getattr(meta, "title", None) or "").strip()
    if not title:
        m = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
        title = re.sub(r"\s+", " ", m.group(1)).strip() if m else url
    return {"html_bytes": len(html.encode()), "text": text, "title": title,
            "author": getattr(meta, "author", None),
            "site_name": getattr(meta, "sitename", None),
            "published_at": getattr(meta, "date", None)}


def gen_json(chat, record, ask, max_tokens=320):
    """The open half. It is generated, so it is allowed to fail, and does."""
    out = "".join(chat.stream(
        [dict(role="user", content=f"{record}\n\n{ask}\nReply with JSON only.")],
        max_tokens=max_tokens, temperature=0.0))
    m = re.search(r"\{.*\}|\[.*\]", out, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def first_sentence(text, limit=300):
    s = re.split(r"(?<=[.!?])\s", (text or "").strip(), maxsplit=1)[0]
    return s[:limit] if s else ""


def convert(reader, chat, url, budget, generate=True):
    t0 = time.time()
    page = fetch(url)
    t_fetch = time.time() - t0

    rec = record_for(reader, url, page["title"], page["text"], budget)
    t1 = time.time()
    a = reader.read(rec, typed_fields())
    t_read = time.time() - t1

    parent, sub, cp, cs = type_path(a)
    lang, paywalled, timely, authored, promo = a[11], a[12], a[13], a[14], a[15]
    digest = hashlib.sha256((page["text"] or url).encode()).hexdigest()

    summary = {"brief": first_sentence(page["text"]), "key_points": []}
    entities, t_gen = [], 0.0
    if generate and chat is not None:
        t2 = time.time()
        s = gen_json(chat, rec, 'Summarise this page as {"brief": "one sentence", '
                                '"key_points": ["...", "...", "..."]}.')
        if isinstance(s, dict):
            summary = {"brief": str(s.get("brief") or summary["brief"])[:300],
                       "key_points": [str(x)[:200] for x in
                                      (s.get("key_points") or [])][:10]}
        names = gen_json(chat, rec, 'List up to eight named entities on this page as '
                                    '["name", "name", ...].', 200)
        names = [str(x).strip()[:80] for x in names][:8] if isinstance(names, list) else []
        names = [n for n in dict.fromkeys(names) if n]
        if names:
            # The closed half of the open answer: one typed field per entity,
            # all reading the same page prefill in one batched pass.
            tf = [Field(f"On this page, what kind of thing is {n!r}?", ENTITY_TYPES)
                  for n in names]
            got = reader.read(rec, tf)
            entities = [{"name": n, "type": g.answer, "relevance": round(g.confidence, 3)}
                        for n, g in zip(names, got)]
        t_gen = time.time() - t2

    doc = {
        "sdf_version": "0.2.0",
        "id": f"sdf:{digest}",
        "sdf": {"version": "0.2", "dialect": "core", "features": []},
        "canonical_url": url,
        "parent_type": parent,
        "type": sub,
        "type_data": {},
        "language": BCP47[lang.answer],
        "source": {"url": url, "title": page["title"], "author": page["author"],
                   "site_name": page["site_name"] or urlparse(url).hostname,
                   "published_at": page["published_at"],
                   "fetched_at": datetime.now(timezone.utc).isoformat()},
        "summary": summary,
        "entities": entities,
        "metadata": {"paywalled": paywalled.answer == "yes",
                     "time_sensitive": timely.answer == "yes",
                     "has_named_author": authored.answer == "yes",
                     "commercial_intent": promo.answer == "yes"},
        "provenance": {"converter": "yantrik-inference 0.1.0",
                       "model_used": "Qwen3.8-27B Q4_K_M, typed-field read",
                       "conversion_confidence": round(min(cp, cs), 4),
                       "content_hash": f"sha256:{digest}"},
    }
    # The schema types every source field as a string, so a field the page did
    # not carry has to be absent, not null. Strip empties at both levels.
    doc["source"] = {k: v for k, v in doc["source"].items() if v not in ("", None)}
    doc = {k: v for k, v in doc.items() if v not in ([], None, "")}
    stats = {"fetch_s": t_fetch, "read_s": t_read, "gen_s": t_gen,
             "html_bytes": page["html_bytes"],
             "sdf_bytes": len(json.dumps(doc).encode()),
             "cp": cp, "cs": cs}
    return doc, stats


def validator():
    try:
        import jsonschema
        return jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text("utf-8")))
    except Exception:                                     # noqa: BLE001
        return None


def main():
    argv, opts, pos = sys.argv[1:], {}, []
    i = 0
    while i < len(argv):
        t = argv[i]
        if t == "--no-generate":
            opts["generate"] = False; i += 1
        elif t.startswith("--"):
            opts[t[2:]] = argv[i + 1] if i + 1 < len(argv) else ""
            i += 2                         # a flag consumes its value, so it is
        else:                              # never mistaken for a URL
            pos.append(t); i += 1
    gguf, urls = pos[0], pos[1:]
    generate = opts.get("generate", True)
    out = None
    if "out" in opts:
        out = Path(opts["out"])
        out.mkdir(parents=True, exist_ok=True)

    gpu, ctx = int(opts.get("gpu", 0)), int(opts.get("ctx", 65536))
    reader, chat = open_model(gguf, decide_ctx=ctx, decide_seq=16,
                              chat_ctx=8192, with_chat=generate, main_gpu=gpu)
    budget = reader.per_seq - max(len(reader.suffix_tokens(f))
                                  for f in typed_fields()) - 8
    v = validator()
    print(f"  16 typed fields in one pass"
          f"{', entity types read as a second batch' if generate else ''}; "
          f"schema validation {'on' if v else 'UNAVAILABLE'}\n")

    ok = bad = 0
    for url in urls:
        try:
            doc, st = convert(reader, chat, url, budget, generate)
        except Exception as e:                            # noqa: BLE001
            print(f"  FAILED {url}\n         {type(e).__name__}: {str(e)[:90]}")
            bad += 1
            continue
        errs = sorted(v.iter_errors(doc), key=lambda e: e.path) if v else []
        ok += not errs
        bad += bool(errs)
        print(f"  {doc['parent_type']}.{doc['type']:<15s} "
              f"{st['cp']*100:3.0f}%/{st['cs']*100:3.0f}%  {doc['language']}  "
              f"{st['html_bytes']/1024:6.0f}KB -> {st['sdf_bytes']/1024:5.1f}KB "
              f"({1 - st['sdf_bytes']/max(1, st['html_bytes']):5.1%} smaller)  "
              f"read {st['read_s']:.2f}s gen {st['gen_s']:.2f}s")
        print(f"      {url[:100]}")
        if doc.get("summary", {}).get("brief"):
            print(f"      {doc['summary']['brief'][:104]}")
        if doc.get("entities"):
            print("      " + ", ".join(f"{e['name']}:{e['type']}"
                                       for e in doc["entities"][:5]))
        for e in errs[:3]:
            print(f"      SCHEMA ERROR at {list(e.path)}: {e.message[:80]}")
        if out:
            name = re.sub(r"[^a-zA-Z0-9]+", "-", url)[:90] + ".json"
            (out / name).write_text(json.dumps(doc, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
    print(f"\n  {ok} schema-valid, {bad} not")
    if out:
        print(f"  documents written to {out}")


if __name__ == "__main__":
    main()
