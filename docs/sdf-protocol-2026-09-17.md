# Converting web pages to SDF Protocol by reading the schema

2026-09-17. Measured on Qwen3.8-27B Q4_K_M, one RTX 3090 Ti capped at 75 °C.

SDF (sdfprotocol.org) compiles a web page once into schema-validated JSON so that
agents stop re-extracting the same page. Most of its schema is closed sets: ten
parent types each with their own subtypes, an eight-value entity type, a language
tag, a handful of booleans. Closed sets are what this library reads rather than
generates, so SDF is a natural test of whether the method earns its keep on
someone else's real problem.

Three results, in order of how much they matter.

## 1. The typed read beats the shipping pipeline on a yardstick neither model wrote

The corpus holds 2344 documents the production pipeline already converted with
GPT-4o-mini. Scoring against those labels would measure agreement, not
correctness, and the labels do not survive inspection: a Wikipedia entry is typed
`reference.academic_paper`, a BBC news story is `reference.unknown`, and two
identical MDN glossary pages come back as `reference.definition` and
`documentation.reference`.

So the labels for this comparison come from URL structure instead
(`sdf_gold.py`). `allrecipes.com/recipe/<id>/<slug>` is a recipe because that is
what the path means on that site. `stackoverflow.com/questions/...` is a question
thread. Rules that cannot be certain return nothing rather than guessing, and a
rule may fix the parent type while leaving the subtype open. 1484 of the 1934
documents with usable text get a label this way.

The record wording was chosen on one random half and scored once on the other.
Dev scored 0.930, test scored 0.928, so nothing was fitted to the test.

Held out, 250 pages:

| | typed read | stored pipeline (GPT-4o-mini) |
|---|---|---|
| parent type | **0.928** | 0.864 |
| full path `parent.subtype` | **0.912** | 0.847 |
| output inside the declared taxonomy | **1.000** | 0.876 |

Majority baseline is 0.516. Median 2.78 s per page for eleven fields in one pass.

The third row is the structural one. 12.4% of the production corpus carries a
type the pipeline's own JSON Schema enum forbids, despite the enum being sent
with every request. A typed read cannot produce an out-of-vocabulary value
because the value is an argmax over the allowed tokens; there is nothing to
validate and nothing to repair. The shipping worker has a `json-repair.ts`
precisely because the other approach fails often enough to need one.

Confidence does the escalation work that the pipeline currently does by calling a
bigger model:

| | share of pages | accuracy |
|---|---|---|
| confidence ≥ 0.97 | 93% | 0.957 |
| below | 7% | 0.529 |

## 2. Two taxonomies ship in the same repository, and the classifier has the wrong one

`spec/protocol.md` §2.2 publishes 52 subtypes. `apps/worker/src/core/prompts/`
`classify.ts` constrains the extractor to a different 45. Only 27 appear in both.

Under `reference` the overlap is **zero**:

| | subtypes |
|---|---|
| published spec | encyclopedia, dictionary, legal, standard, specification |
| shipping classifier | academic_paper, legal_document, medical_info, patent, recipe |

This is not cosmetic. Asked the same pages under each vocabulary:

| page | classifier vocabulary | spec vocabulary |
|---|---|---|
| Wikipedia, Ancient Egypt | `reference.academic_paper` 88% | `reference.encyclopedia` **100%** |
| Wikipedia, Photosynthesis | `reference.academic_paper` 82% | `reference.encyclopedia` **100%** |
| MDN glossary entry | `documentation.guide` 94% | `documentation.reference` 73% |
| Kubernetes Deployments | `documentation.guide` 89% | `documentation.guide` 73% |
| PyPI httpx | `code.package` 100% | `code.package` 100% |

A Wikipedia article is an encyclopedia entry. The published protocol has that
word. The classifier does not, so every encyclopedia page in the corpus was
forced into the nearest wrong answer, confidently. That also explains the 12.4%:
the model keeps reaching for spec vocabulary it was never given, and the enum
rejects it.

Three things follow for the SDF project, none of which need this library:

1. Reconcile `classify.ts` with `spec/protocol.md`, or say which one is
   normative. Right now consumers validating against the published schema and
   producers constrained by the worker disagree about 25 of 52 subtypes.
2. The 290 out-of-taxonomy documents in the store are not all model errors. Some
   are the model being right in the published vocabulary.
3. `article` and `code` are nearly aligned; `reference`, `commerce` and `data`
   are where the drift is worst.

## 3. Generation is four times the cost, and it is the part that can be invalid

The converter (`sdf_convert.py`) splits the document by what kind of answer each
field takes:

- **read**: parent type, subtype, language, paywalled, time-sensitive, has an
  author, commercial intent, and every entity's type
- **generate**: the brief, the key points, the entity *names*

Entity type is the clearest case. The schema fixes it to eight values, so a
generated entity type can be wrong in a way no parser catches. Here the names are
generated and their types are read as one batched pass over the same page
prefill.

Eight live pages converted end to end, all eight validating against
`sdf-document-0.2.schema.json`:

| page | type | read | generate |
|---|---|---|---|
| Guardian arts feature | `article.review` 94%/40% | 15.4 s | 75.7 s |
| Guardian opinion column | `article.opinion` 100%/100% | 16.7 s | 36.9 s |
| Cloudflare blog | `article.blog` 97%/49% | 6.3 s | 20.9 s |
| Wikipedia | `reference.academic_paper` 100%/88% | 6.9 s | 24.0 s |
| Kubernetes docs | `documentation.guide` 100%/90% | 6.8 s | 22.7 s |
| PyPI httpx | `code.package` 100%/100% | 4.5 s | 15.7 s |
| GitHub encode/httpx | `code.repository` 100%/100% | 3.7 s | 13.2 s |
| Hacker News thread | `discussion.comment_thread` 100%/89% | 2.6 s | 6.0 s |

Generation costs roughly four times the read on every page. The two low subtype
confidences are the two cases worth arguing about: a photo essay is arguably not
a review, and a product announcement is arguably not a blog post. The confidence
found them without being told to.

Size reduction runs 98.7–99.8%, which is consistent with the protocol's own
claim.

## What broke along the way

- **`llama_decode` asserts when a batch exceeds `n_batch`.** A page-sized record
  aborted the process rather than raising. Both the field reader and the chat
  engine now prefill in `n_batch`-sized pieces; the pieces are one sequence in
  order with no logits wanted, so the resulting cache is exactly what one large
  batch would have produced.
- **A character budget is wrong for a multilingual corpus.** 9000 characters is
  2400 tokens of English and 4400 of Arabic, so a fixed character cut overflowed
  the sequence on the hardest pages. The record is now trimmed in tokens.
- **The per-sequence limit is real even though the prefix is shared.**
  `llama_memory_seq_cp` tags cells rather than copying them, so it seemed the
  limit should be total cells against `n_ctx` rather than record size against
  `n_ctx / n_seq`. `probe_cells.py` tested that directly: a record 1.1× the
  per-sequence budget is refused, and 2×, 4× and 8× likewise. The conservative
  check was correct and was left alone. Sizing rule: `n_ctx` must be at least
  the longest record times the number of fields.
- **Wikipedia answers a browser-spoofing client with "please respect our robot
  policy".** The converter now identifies itself honestly, reads `robots.txt`
  before fetching, and uses each site's sanctioned endpoint where one exists.
  Stack Overflow still refuses non-browser clients and is left alone.
- **The schema types every `source` field as a string**, so a field the page did
  not carry has to be absent rather than null. Four of the first eight documents
  failed validation on a null author before this was fixed, which is the argument
  for validating output rather than asserting it is valid by construction.

## Files

`probe_sdf.py` scores the taxonomy read against URL gold with a dev/test split.
`sdf_gold.py` holds the URL rules. `sdf_convert.py` converts live URLs to SDF
documents and validates them. `probe_taxonomy.py` compares the two vocabularies.
`probe_cells.py` tests whether the per-sequence limit is real.

---

## Addendum: the same task against the real Jev API

Added 2026-09-17, after installing TypeSafe's skill and getting API access.

Everything above is a local reproduction of the mechanism TypeSafe's Jev exposes.
`probe_jev.py` runs the real thing on exactly this task, with the same question
set, the same category descriptions, and the same URL-derived gold, so the
comparison is like for like. The eleven questions go in one request, which is what
TypeSafe calls speculative fan-out: each subtype question states its own premise,
and code takes the one belonging to the chosen parent.

150 held-out pages, `jev-1.13.0`:

| | Jev | local 27B | the SDF pipeline's stored labels |
|---|---|---|---|
| parent type | **0.940** | 0.928 | 0.847 |
| full `parent.subtype` path | 0.902 | **0.912** | 0.833 |
| seconds per page, 11 questions | **0.17** | 2.78 | — |

Majority baseline 0.513. Jev used 2532 input tokens per page.

Two things worth saying plainly.

**On accuracy the hosted model and the local reproduction are indistinguishable.**
0.940 against 0.928 is two pages out of 150, and on the full path the local
reproduction is nominally ahead. Neither gap means anything at this sample size.

**On speed they are not close.** 0.17 s against 2.78 s is about sixteen times, and
that is a 27B model on one consumer GPU against a purpose-built hosted service.
The mechanism reproduces. The engineering does not.

Jev's calibration is good on this task:

| confidence | pages | correct |
|---|---|---|
| 0.97 and above | 118 | 0.96 |
| 0.85 to 0.97 | 23 | 0.96 |
| 0.60 to 0.85 | 7 | 0.86 |
| below 0.60 | 2 | 0.00 |

**The errors are the same errors.** `documentation` misread as `reference`
dominates for Jev, exactly as it does for the local reader. Two independent
implementations failing on the same boundary is evidence the boundary itself is
underdefined, not that either model is weak. It is the same boundary where the
published spec and the shipping classifier disagree most.

One caveat on matching: Jev received 6000 characters of page text and the local
reader about 4800, since the local budget is set in tokens against a per-sequence
limit. Close, not identical.
