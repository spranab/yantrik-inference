"""Gold values for extraction, taken from what publishers declare about themselves.

Scoring an extractor against another model's output measures agreement. Most
pages worth converting already carry the answer in machine-readable form:
schema.org JSON-LD, OpenGraph tags, or the HTML `<time datetime>` attribute, all
written by the publisher for search engines. That is a ground truth no model
produced and neither side of the comparison can see, because the extractor is
given the rendered text with the markup stripped.

Returns the fields it can find, and says nothing where the page declares nothing.
"""
from __future__ import annotations

import html as _html
import json
import re

MONTH_NAMES = ("january february march april may june july august september october november december".split())

INTERESTING = ("headline", "author", "publish_date", "section", "price")


def _texts(node, key):
    """schema.org lets almost every field be a string, an object or a list."""
    v = node.get(key)
    out = []
    for item in v if isinstance(v, list) else [v]:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            n = item.get("name")
            # schema.org nodes often carry only an @id, which is a URL pointing
            # at another node. Taking it as the value made every TechCrunch
            # author "https://techcrunch.com/#/schema/person/image/7eb1...",
            # which no extractor could ever match. An @id is a reference, not a
            # name, so a node without a name contributes nothing.
            if isinstance(n, str) and not n.startswith(("http://", "https://")):
                out.append(n)
    return [x.strip() for x in out if x and x.strip()]


def _walk(node, found):
    if isinstance(node, list):
        for x in node:
            _walk(x, found)
        return
    if not isinstance(node, dict):
        return
    for k in ("headline", "name", "title"):
        if k in node and "headline" not in found:
            t = _texts(node, k)
            if t and len(t[0]) > 10:
                found["headline"] = t[0]
            break
    if "author" in node and "author" not in found:
        a = _texts(node, "author")
        if a:
            found["author"] = a[0]
    for k in ("datePublished", "dateCreated", "uploadDate"):
        if k in node and "publish_date" not in found:
            t = _texts(node, k)
            if t:
                found["publish_date"] = t[0]
            break
    if "articleSection" in node and "section" not in found:
        t = _texts(node, "articleSection")
        if t:
            found["section"] = t[0]
    for k in ("offers", "aggregateOffer", "mainEntity", "@graph", "itemListElement"):
        if k in node:
            _walk(node[k], found)
    if "price" in node and "price" not in found:
        p = node["price"]
        if isinstance(p, (str, int, float)):
            found["price"] = str(p).strip()


def from_html(html: str) -> dict:
    found: dict = {}
    for m in re.finditer(
            r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
            html, re.S | re.I):
        raw = m.group(1).strip()
        try:
            _walk(json.loads(raw), found)
        except json.JSONDecodeError:
            continue

    # OpenGraph and standard meta tags fill gaps JSON-LD left.
    def meta(*names):
        for n in names:
            m = re.search(
                r'<meta[^>]+(?:property|name)=["\']' + re.escape(n) +
                r'["\'][^>]+content=["\'](.*?)["\']', html, re.I | re.S)
            if not m:
                m = re.search(
                    r'<meta[^>]+content=["\'](.*?)["\'][^>]+(?:property|name)=["\']'
                    + re.escape(n) + r'["\']', html, re.I | re.S)
            if m:
                return _html.unescape(m.group(1)).strip()
        return None

    found.setdefault("headline", meta("og:title", "twitter:title"))
    found.setdefault("author", meta("author", "article:author", "parsely-author"))
    found.setdefault("publish_date", meta("article:published_time", "date",
                                          "parsely-pub-date", "datePublished"))
    found.setdefault("section", meta("article:section", "parsely-section"))
    found.setdefault("price", meta("product:price:amount", "og:price:amount"))

    if not found.get("publish_date"):
        m = re.search(r'<time[^>]+datetime=["\']([^"\']+)["\']', html, re.I)
        if m:
            found["publish_date"] = m.group(1).strip()

    return {k: v for k, v in found.items()
            if k in INTERESTING and isinstance(v, str) and v.strip()
            and not (k in ("author", "headline", "section")
                     and v.strip().startswith(("http://", "https://")))}


def ymd(s: str):
    """(year, month, day) from an ISO stamp or a written date, or None.

    The first version of this only matched year-first strings, so a page that
    rendered "September 11, 2026" was scored as failing to contain a date its
    own markup declared as 2026-09-11. That was a bug in the scorer, and it made
    a working extraction look like a dead end.
    """
    if not s:
        return None
    t = s.strip().lower()
    m = re.search(r"(\d{4})\D(\d{1,2})\D(\d{1,2})", t)
    if m:
        y, mo, d = (int(x) for x in m.groups())
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return y, mo, d
    m = re.search(r"([a-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", t)
    if not m:
        m2 = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]{3,9})\.?,?\s+(\d{4})", t)
        if m2:
            d, name, y = m2.group(1), m2.group(2), m2.group(3)
            m = None
        else:
            name = None
    else:
        name, d, y = m.group(1), m.group(2), m.group(3)
    if name:
        for i, full in enumerate(MONTH_NAMES, 1):
            if full.startswith(name[:3]):
                return int(y), i, int(d)
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", t)
    if m:
        a, b, y = (int(x) for x in m.groups())
        return (y, b, a) if a > 12 else (y, a, b)
    return None


def same(a: str, b: str, kind: str) -> bool:
    """Compare a predicted value with the declared one, allowing for formatting.

    Dates are compared on the calendar date only, because a page may render
    "September 11, 2026" while its markup says 2026-09-11T13:00:00Z and those
    are the same fact. Text is compared on lowercase alphanumerics, because a
    declared headline routinely differs from the rendered one by a trailing site
    name or a typographic apostrophe.
    """
    if not a or not b:
        return False
    if kind == "publish_date":
        da, db = ymd(a), ymd(b)
        return bool(da and db and da == db)
    na, nb = _norm(a), _norm(b)
    if not na or not nb or min(len(na), len(nb)) < 4:
        return False
    return na == nb or na in nb or nb in na


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())
