"""URL-derived gold labels for the SDF corpus, assigned by publishing convention.

The corpus's stored types came from an LLM, so scoring a model against them
measures agreement, not correctness — and inspection shows the stored labels are
unreliable: a Wikipedia article is typed `reference.academic_paper`, a BBC news
story is `reference.unknown`, and two identical MDN glossary pages get
`reference.definition` and `documentation.reference`.

These rules use only the URL. `allrecipes.com/recipe/<id>/<slug>` is a recipe
because that is what the path segment means on that site, not because a model
said so. Rules that cannot be certain return None rather than guessing, and a
rule may fix the parent while leaving the subtype open.

Returns (parent, subtype | None) or None.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

DOC_HOSTS = {"kubernetes.io", "docs.cloud.google.com", "docs.stripe.com",
             "docs.aws.amazon.com", "docs.docker.com", "nextjs.org",
             "docs.github.com", "porkbun.com", "www.elastic.co"}
NEWS_HOSTS = {"techcrunch.com", "arstechnica.com", "www.technologyreview.com",
              "www.infoq.com"}
CONF_HOSTS = {"www.dotnetconf.net", "www.gophercon.com", "rustconf.com",
              "2025.djangocon.us", "icml.cc", "sxsw.com", "build.microsoft.com"}


def gold(url: str):
    u = urlparse(url)
    h, p = (u.hostname or "").lower(), u.path

    if h == "www.bbc.com":
        # /news/... and /<lang>/articles/... are stories; /sport/ too.
        if re.search(r"/(news|articles)/", p) or p.startswith("/sport/"):
            return ("article", "news")
        return None
    if h == "www.theguardian.com":
        if re.search(r"/\d{4}/[a-z]{3}/\d{2}/", p):
            return ("article", "news")
        return None
    if h in NEWS_HOSTS:
        return ("article", None)          # news vs blog is a judgement call
    if h == "blog.cloudflare.com":
        return ("article", None)

    if h == "stackoverflow.com" and p.startswith("/questions/"):
        return ("discussion", "q_and_a")
    if h == "old.reddit.com" and "/comments/" in p:
        return ("discussion", "forum_thread")

    if h == "www.allrecipes.com" and p.startswith("/recipe/"):
        return ("reference", "recipe")
    if h == "en.wikipedia.org" and p.startswith("/wiki/"):
        return ("reference", None)        # no encyclopedia subtype exists
    if h == "www.sciencedirect.com" and "/science/article/" in p:
        return ("reference", "academic_paper")
    if h == "www.nature.com" and re.search(r"/articles/", p):
        return ("reference", "academic_paper")
    if h == "www.investopedia.com" and p.startswith("/terms/"):
        return ("reference", None)        # a dictionary entry; no such subtype

    if h in DOC_HOSTS and ("/docs" in p or "/api" in p or "/documentation" in p):
        return ("documentation", None)
    if h == "developer.mozilla.org" and "/docs/" in p:
        return ("documentation", None)

    if h == "pypi.org" and p.startswith("/project/"):
        return ("code", "package")
    if h == "github.com" and re.fullmatch(r"/[^/]+/[^/]+/?", p):
        return ("code", "repository")

    if h == "www.ted.com" and p.startswith("/talks/"):
        return ("media", "video")
    if h in CONF_HOSTS:
        return ("event", "conference")
    return None
