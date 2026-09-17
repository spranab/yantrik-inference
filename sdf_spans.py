"""Turn extraction into a closed choice by harvesting candidates from the page.

A typed read answers questions with a fixed option list. A headline is not a
fixed option list, which looks like the end of the idea for extraction. It is
not, because the answer to "what is the headline" is always a span that is
already printed on the page.

So harvest the candidate spans deterministically, with regular expressions over
the rendered text, and let the model pick one. Two properties follow:

  the value cannot be hallucinated, because every option is a literal span
  copied from the source, and "none" is always available

  the choice is a closed set, so it reads from the logits like any other field

Candidates are labelled A, B, C so the options differ in their first token. Two
dates both starting "2026" would alias; the letters cannot.

Nothing here reads the HTML markup. The gold values in sdf_ld.py come from
JSON-LD and meta tags, so harvesting from markup would be copying the answer.
These functions see only what a reader sees.
"""
from __future__ import annotations

import html as _html
import re
from typing import List

LETTERS = tuple("ABCDEFGHIJKLMNOPQRST")


def visible_text(html: str, limit=20000) -> str:
    """Everything a reader sees, which is more than the article body.

    trafilatura in precision mode is built to return the ARTICLE, so it drops
    the masthead, the byline and the dateline. Measured on live pages, that put
    the headline, author and date outside the text on every one: the candidate
    lists could not contain the answer because the text did not.

    So candidates come from the rendered page instead. <head> is removed
    entirely, which keeps this honest: <title>, the OpenGraph tags and the
    JSON-LD block are the gold in sdf_ld.py, and none of them survive here.
    """
    h = re.sub(r"(?is)<head\b.*?</head>", " ", html or "")
    for tag in ("script", "style", "noscript", "svg", "template"):
        h = re.sub(rf"(?is)<{tag}\b.*?</{tag}>", " ", h)
    h = re.sub(r"(?is)<!--.*?-->", " ", h)
    h = re.sub(r"(?i)<(?:br|/p|/div|/li|/h[1-6]|/tr|/section)\s*/?>", "\n", h)
    h = re.sub(r"(?s)<[^>]+>", " ", h)
    h = _html.unescape(h)
    lines = [re.sub(r"[ \t]+", " ", l).strip() for l in h.splitlines()]
    return "\n".join(l for l in lines if l)[:limit]


MONTHS = ("January|February|March|April|May|June|July|August|September|October|"
          "November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec")

DATE_RE = re.compile(
    r"\b(?:"
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?"          # 2026-09-16
    r"|(?:" + MONTHS + r")\.?\s+\d{1,2},?\s+\d{4}"                 # September 16, 2026
    r"|\d{1,2}\s+(?:" + MONTHS + r")\.?\s+\d{4}"                   # 16 September 2026
    r"|\d{1,2}/\d{1,2}/\d{4}"                                      # 16/09/2026
    r")\b", re.I)

PRICE_RE = re.compile(r"(?:[$£€¥]\s?\d[\d,]*(?:\.\d{2})?"
                      r"|\b\d[\d,]*(?:\.\d{2})?\s?(?:USD|EUR|GBP|INR)\b)")

BYLINE_RE = re.compile(r"^\s*(?:By|By:|Written by|Author:)\s+(.{2,60}?)\s*$",
                       re.I | re.M)

# Two to four capitalised words, the shape of a person's name in running text.
NAME_RE = re.compile(r"\b([A-Z][a-z'’\-]{1,15}(?:\s+[A-Z][a-z'’\-]{1,15}){1,3})\b")

STOP_NAMES = {"the", "this", "that", "new", "united states", "privacy policy",
              "terms of service", "sign in", "read more", "getting started"}


def _clean(s: str, limit=110) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip(" \t-–—|·•")
    return s[:limit]


def _dedupe(items: List[str], cap: int) -> List[str]:
    out, seen = [], set()
    for x in items:
        x = _clean(x)
        k = re.sub(r"[^a-z0-9]+", "", x.lower())
        if not k or k in seen or len(x) < 2:
            continue
        seen.add(k)
        out.append(x)
        if len(out) >= cap:
            break
    return out


NAV = re.compile(
    r"^(skip to|all categories|sign in|log in|subscribe|menu|search|share|"
    r"follow us|newsletter|cookie|accept|privacy|terms|contact|about us|"
    r"back to|home$|more from|related|advertisement|toggle)", re.I)


def _content_lines(text: str):
    """Lines that look like page content rather than furniture.

    The first attempt took the first 25 lines, which on a modern site is the
    navigation bar: "Skip to content", "All Categories", a language switcher.
    The headline sits below that, so it never reached the candidate list.
    """
    out = []
    for l in (text or "").splitlines():
        l = l.strip()
        if not l or NAV.match(l) or l.count("|") > 2:
            continue
        out.append(l)
    return out


def _slug_words(url: str):
    """Content words from the URL path.

    A news URL almost always carries the headline as a slug. The URL is an input
    to the converter, not markup, so using it to rank candidates is fair, and it
    separates the real headline from the promotional banners a site puts above
    it. Without this, TechCrunch's "Save up to $300 on Disrupt" outranked the
    actual title on every article.
    """
    from urllib.parse import urlparse
    tail = urlparse(url or "").path.rstrip("/").split("/")[-1]
    return {w for w in re.split(r"[-_]+", tail.lower()) if len(w) > 3}


def headlines(text: str, url: str = "", cap=12) -> List[str]:
    """Title-shaped lines, ranked by agreement with the URL slug."""
    lines = _content_lines(text)
    slug = _slug_words(url)
    scored = []
    for i, l in enumerate(lines[:150]):
        if not (18 <= len(l) <= 170) or len(l.split()) < 4:
            continue
        if l.endswith((".", ":", ";", ",")) and len(l.split()) > 14:
            continue
        words = {w for w in re.split(r"[^a-z0-9]+", l.lower()) if len(w) > 3}
        overlap = len(words & slug) / max(1, len(slug))
        scored.append((-overlap * 100 + i / 10.0, l))
    scored.sort()
    return _dedupe([l for _, l in scored], cap)


def authors(text: str, cap=12) -> List[str]:
    lines = _content_lines(text)
    head = "\n".join(lines[:60])
    out = [m.group(1) for m in BYLINE_RE.finditer("\n".join(lines))]
    out += [m.group(1) for m in re.finditer(r"\bBy\s+([A-Z][^,\n]{2,40}?)"
                                            r"(?:\s*[,|]|\s+on\b|$)", head)]
    out += [m.group(1) for m in NAME_RE.finditer(head)]
    out = [x.strip() for x in out]
    out = [x for x in out if x.lower() not in STOP_NAMES
           and not re.match(r"(?:" + MONTHS + r")\b", x, re.I)
           and not NAV.match(x)]
    return _dedupe(out, cap)


def dates(text: str, cap=10) -> List[str]:
    return _dedupe([m.group(0) for m in DATE_RE.finditer(text or "")], cap)


def prices(text: str, cap=8) -> List[str]:
    return _dedupe([m.group(0) for m in PRICE_RE.finditer(text or "")], cap)


def sections(url: str, text: str, cap=8) -> List[str]:
    from urllib.parse import urlparse
    segs = [s for s in urlparse(url).path.split("/")
            if s and not s.isdigit() and len(s) < 24 and "-" not in s]
    return _dedupe(segs + [l.strip() for l in (text or "").splitlines()[:6]], cap)


HARVEST = {"headline": lambda url, text: headlines(text, url),
           "author": lambda url, text: authors(text),
           "publish_date": lambda url, text: dates(text),
           "price": lambda url, text: prices(text),
           "section": lambda url, text: sections(url, text)}


def menu(url: str, text: str, fields=("headline", "author", "publish_date")):
    """Candidate lists for each field, as {field: [span, ...]}."""
    return {f: HARVEST[f](url, text) for f in fields}


def render(cands: dict) -> str:
    """The candidate lists as they appear in the shared prefix."""
    out = ["CANDIDATE SPANS COPIED FROM THIS PAGE",
           "Each list holds text that literally appears above. Choose the letter "
           "of the correct one, or 'none' if the page does not state it."]
    for field, items in cands.items():
        out.append(f"\n{field} candidates:")
        if not items:
            out.append("  (none found)")
        for letter, item in zip(LETTERS, items):
            out.append(f"  {letter}) {item}")
    return "\n".join(out)


def options(items) -> tuple:
    return tuple(LETTERS[:len(items)]) + ("none",)


def resolve(answer: str, items) -> str | None:
    if answer == "none" or answer not in LETTERS:
        return None
    i = LETTERS.index(answer)
    return items[i] if i < len(items) else None
