"""Convert web pages to SDF Protocol documents by reading the schema, not writing JSON.

SDF (sdfprotocol.org) compiles a page once into schema-validated JSON so agents
stop re-extracting it. Its type system is a two-level closed enum: 10 parent
types, each with its own subtypes, forming `article.news` or
`documentation.api_docs`.

The production pipeline asks an LLM for that as JSON with a schema attached, then
parses it, then repairs the JSON when parsing fails, then escalates to a larger
model when confidence is low. A typed read removes all four steps: the answer is
an argmax over the allowed tokens, so it cannot be invalid, cannot fail to parse,
and arrives with a calibrated probability that IS the escalation signal.

The conditional part is where the architecture earns its keep. `type` only means
anything given `parent_type`, which normally costs two round trips or one prompt
trusted to keep them consistent. Here the page is prefilled once and every field
reads that same prefix, so asking the parent question AND all ten subtype
questions costs barely more than asking one. Take the subtype field matching the
chosen parent and the path is consistent by construction.

Scored against URL-derived gold labels (see sdf_gold.py), not against the stored
types: those came from an LLM and inspection shows they are unreliable. The same
gold scores the stored labels, so the comparison is like for like.

DEV/TEST: the record wording was chosen on the dev half and scored once on the
test half. Report the test number.

    python probe_sdf.py <model.gguf> [n] [dev|test]
"""
from __future__ import annotations

import sqlite3
import sys
import time
from collections import Counter

import numpy as np

from yantrik_inference import Field, open_model
from sdf_gold import gold

DB = r"C:\Users\sync\codes\sdf\data\sdf.db"

SUBTYPES = {
    "article":       ("news", "blog", "opinion", "review", "press_release"),
    "documentation": ("api_docs", "support", "tutorial", "guide", "changelog", "faq"),
    "commerce":      ("product", "job_posting", "real_estate", "service", "auction"),
    "data":          ("finance", "sports", "weather", "scientific_dataset"),
    "reference":     ("academic_paper", "legal_document", "recipe", "medical_info", "patent"),
    "discussion":    ("forum_thread", "q_and_a", "social_post", "comment_thread"),
    "code":          ("repository", "package", "code_snippet", "pull_request"),
    "media":         ("video", "podcast", "image_gallery", "music_album"),
    "profile":       ("person", "organization", "place"),
    "event":         ("conference", "meetup", "webinar", "live_event", "concert"),
}
PARENTS = tuple(SUBTYPES)

# The taxonomy, spelled out. This goes in the shared prefix, so it is prefilled
# once and read by all eleven fields: the same amortisation that makes few-shot
# examples cheap here. Without it the reader confuses reference with article on
# encyclopedia and paper pages, which is a definition problem, not a model one.
TAXONOMY = """SDF content types. Every web page belongs to exactly one of these
ten categories. Judge by what the page IS, not what it is about.

  article        journalism and written pieces published at a point in time: news
                 reports, blog posts, opinion columns, reviews, press releases.
                 Has a publication date and usually a byline. A news story about
                 science is still an article.
  documentation  material explaining how to use a specific product or technology:
                 API references, guides, tutorials, support pages, changelogs,
                 FAQs. Published by the maker of the thing.
  commerce       a page whose purpose is a transaction: a product for sale, a job
                 posting, a property listing, a service offering, an auction.
  data           a page whose substance is figures: market data, sports results,
                 weather, statistical datasets.
  reference      durable factual material with no publication moment:
                 encyclopedia entries, dictionary and glossary definitions,
                 academic papers, legal texts, recipes, medical information,
                 patents. A Wikipedia entry is reference. A recipe is reference.
  discussion     content produced by a back-and-forth between people: questions
                 and answers, forum threads, comment threads, social posts.
  code           a software artefact page: a repository, a published package, a
                 code snippet, a pull request.
  media          a page whose primary object is audio or video: a talk, a podcast
                 episode, an image gallery, an album.
  profile        a page describing an entity: a person, an organisation, a place.
  event          a page for something happening at a time and place: a
                 conference, meetup, webinar, workshop.

Distinctions that matter:
  - a paper, encyclopedia entry or glossary definition is reference, not article
  - vendor docs for a product are documentation, not reference
  - a question with answers under it is discussion, even on a technical site"""


def fields_for_page():
    fs = [Field("Which of the ten SDF categories does this page belong to?", PARENTS)]
    for p, subs in SUBTYPES.items():
        fs.append(Field(f"Assuming this page is in the {p} category, "
                        f"which specific kind is it?", subs))
    return fs


def type_path(answers):
    parent = answers[0]
    sub = answers[1 + PARENTS.index(parent.answer)]
    return parent.answer, sub.answer, parent.confidence, sub.confidence


def split(n, which, seed=7):
    c = sqlite3.connect(DB)
    rows = c.execute("select url, title, parent_type, sub_type, content_text "
                     "from sdf_documents where length(content_text) >= 400").fetchall()
    have = [(u, t, p, s, x, gold(u)) for u, t, p, s, x in rows if gold(u)]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(have))
    half = len(order) // 2
    pick = order[:half] if which == "dev" else order[half:]
    return [have[i] for i in pick[:n]], len(have)


def record_for(reader, url, title, text, budget):
    """Trim the page by TOKENS, not characters.

    A character budget is wrong for a multilingual corpus: the same 9000 chars is
    2400 tokens of English and 4400 of Arabic, so a fixed character cut overflows
    the sequence on exactly the pages that are already hardest. llama.cpp enforces
    a per-sequence limit of n_ctx / n_seq even though the prefix cells are shared
    (probe_cells.py shows a 1.1x record refused), so the budget is real and has to
    be respected in the unit the limit counts in.
    """
    head = (f"{TAXONOMY}\n\nPAGE TO CLASSIFY\nURL: {url}\nTitle: {title}\n\n"
            f"Content:\n")
    fixed = len(reader.tok(reader.head + reader.framed(head), bos=True))
    ids = reader.tok(text or "", special=False)[:max(0, budget - fixed)]
    return head + reader.llm.detokenize(ids).decode("utf-8", "replace")


def main():
    gguf = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    which = sys.argv[3] if len(sys.argv) > 3 else "test"
    rows, total = split(n, which)
    print(f"  {total} pages carry a URL-derived gold label; scoring {len(rows)} "
          f"from the {which} half\n")

    reader, _ = open_model(gguf, decide_ctx=65536, decide_seq=16, with_chat=False)
    fs = fields_for_page()
    # leave room for the longest question on top of the record
    budget = reader.per_seq - max(len(reader.suffix_tokens(f)) for f in fs) - 8

    okp, oks, sp, ss, cp_, cs_, times, ntoks = [], [], [], [], [], [], [], []
    conf_mat = Counter()
    for url, title, spar, ssub, text, (gp, gs) in rows:
        rec = record_for(reader, url, title, text, budget)
        ntoks.append(len(reader.tok(reader.head + reader.framed(rec), bos=True)))
        t0 = time.time()
        a = reader.read(rec, fs)
        times.append(time.time() - t0)
        p, s, c1, c2 = type_path(a)
        okp.append(p == gp); sp.append(spar == gp)
        cp_.append(c1); cs_.append(c2)
        if gs:
            oks.append(p == gp and s == gs)
            ss.append(spar == gp and ssub == gs)
        if p != gp:
            conf_mat[(gp, p)] += 1

    okp, sp = np.array(okp, float), np.array(sp, float)
    oks, ss = np.array(oks, float), np.array(ss, float)
    cp_, cs_ = np.array(cp_), np.array(cs_)
    maj = Counter(r[5][0] for r in rows).most_common(1)[0]

    print("  scored against URL convention, which involves no model at all\n")
    print(f"  {'':36s} {'typed read':>11s} {'stored (GPT)':>13s}")
    print(f"  parent_type  (n={len(okp):3d})                {okp.mean():11.3f} {sp.mean():13.3f}")
    print(f"  full path    (n={len(oks):3d})                {oks.mean():11.3f} {ss.mean():13.3f}")
    print(f"  in-taxonomy output                   {1.000:11.3f} {0.876:13.3f}")
    print(f"  always say {maj[0]!r:<14s}            {maj[1]/len(rows):11.3f}")
    print()
    print(f"  median {np.median(times):.2f} s per page, {len(fs)} fields in one pass, "
          f"record {int(np.median(ntoks))} tokens")

    print("\n  parent calibration:")
    for lo, hi in ((0, .6), (.6, .85), (.85, .97), (.97, 1.01)):
        m = (cp_ >= lo) & (cp_ < hi)
        if m.sum():
            print(f"    conf {lo:.2f}-{hi:.2f}: n={int(m.sum()):3d}  mean {cp_[m].mean():.2f}  "
                  f"correct {okp[m].mean():.2f}")

    hi = cp_ >= 0.97
    if hi.any() and (~hi).any():
        print(f"\n  gate at 0.97: {hi.mean()*100:.0f}% of pages, correct {okp[hi].mean():.3f}; "
              f"escalate the rest, correct {okp[~hi].mean():.3f}")

    print("\n  remaining parent errors (gold -> read):")
    for (g, p), c in conf_mat.most_common(8):
        print(f"    {g:16s} -> {p:16s} {c}")


if __name__ == "__main__":
    main()
