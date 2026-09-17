"""Two taxonomies ship in the same project, and the classifier has the wrong one.

`spec/protocol.md` §2.2 publishes the SDF v0.2 type taxonomy: 52 subtypes.
`apps/worker/src/core/prompts/classify.ts` constrains the extractor to a
different list of 45. Only 27 appear in both, and under `reference` the overlap
is zero — the spec says encyclopedia, dictionary, legal, standard,
specification; the classifier allows academic_paper, legal_document,
medical_info, patent, recipe.

The consequence is visible in the stored corpus: 12.4% of documents carry a type
the classifier's own enum forbids, because the model keeps reaching for the
published vocabulary it was not given.

This asks the same pages under each taxonomy. If a page that is forced into a
poor answer under one becomes obvious under the other, the classifier is wrong
about its vocabulary, not about the page.

    python probe_taxonomy.py <model.gguf>
"""
from __future__ import annotations

import sys

from yantrik_inference import Field, open_model
from probe_sdf import PARENTS, SUBTYPES, record_for
from sdf_convert import fetch

SPEC_SUBTYPES = {
    "article":       ("news", "blog", "opinion", "review", "analysis", "press_release"),
    "documentation": ("api_docs", "guide", "tutorial", "reference", "support", "changelog"),
    "commerce":      ("product", "service", "listing", "comparison", "deal"),
    "data":          ("finance", "scientific_dataset", "statistics", "report", "survey"),
    "reference":     ("encyclopedia", "dictionary", "legal", "standard", "specification"),
    "discussion":    ("q_and_a", "forum_thread", "comment_thread", "debate", "poll"),
    "code":          ("repository", "snippet", "package", "gist", "notebook"),
    "media":         ("video", "podcast", "image_gallery", "livestream", "playlist"),
    "profile":       ("person", "organization", "team", "project", "community"),
    "event":         ("conference", "meetup", "webinar", "workshop", "hackathon"),
}

PAGES = [
    "https://en.wikipedia.org/wiki/Ancient_Egypt",
    "https://en.wikipedia.org/wiki/Photosynthesis",
    "https://developer.mozilla.org/en-US/docs/Glossary/Hoisting",
    "https://kubernetes.io/docs/concepts/workloads/controllers/deployment/",
    "https://pypi.org/project/httpx/",
]


def ask(reader, rec, table):
    fs = [Field("Which of the ten SDF categories does this page belong to?", PARENTS)]
    for p, subs in table.items():
        fs.append(Field(f"Assuming this page is in the {p} category, "
                        f"which specific kind is it?", subs))
    a = reader.read(rec, fs)
    par = a[0]
    sub = a[1 + PARENTS.index(par.answer)]
    return f"{par.answer}.{sub.answer}", sub.confidence


def main():
    reader, _ = open_model(sys.argv[1], decide_ctx=65536, decide_seq=16, with_chat=False)
    budget = reader.per_seq - 60
    print(f"  {'page':46s} {'worker classify.ts':>28s} {'spec protocol.md':>28s}\n")
    for url in PAGES:
        try:
            page = fetch(url)
        except Exception as e:                            # noqa: BLE001
            print(f"  {url[:46]:46s} fetch failed: {type(e).__name__}")
            continue
        rec = record_for(reader, url, page["title"], page["text"], budget)
        w, wc = ask(reader, rec, SUBTYPES)
        s, sc = ask(reader, rec, SPEC_SUBTYPES)
        print(f"  {url[8:54]:46s} {w:>22s} {wc*100:3.0f}% {s:>22s} {sc*100:3.0f}%")
    print("\n  Where the two disagree, the spec column is the published protocol")
    print("  and the worker column is what the shipping classifier can emit.")


if __name__ == "__main__":
    main()
