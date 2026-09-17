# Does an SDF document substitute for the page? Measured: not yet

2026-09-17. Qwen3.8-27B Q4_K_M, one RTX 3090 Ti capped at 75 °C.

SDF exists so that a page is converted once and nobody has to read it again. The
test that follows from that is not whether a headline matches a meta tag. It is
whether an agent asking something about a page gets the same answer from the
document as it would from the page.

`probe_summary.py` generates multiple-choice questions about a page, then answers
them four ways: from the full page, from the SDF document, from the **same number
of tokens of raw page text**, and from the URL and title alone. The reference is
what the model answers holding the whole page, so this measures preservation
rather than truth. The third condition is the one that matters. A document that
only matches plain truncation to the same size has earned nothing.

## The result

18 pages interleaved across eight feeds, 108 questions:

| context given | tokens | agreement with the full page |
|---|---|---|
| full page | 1865 | 1.000 |
| SDF document | 1208 | 0.870 |
| **same-size raw page text** | 1177 | **0.935** |
| url and title only | 507 | 0.815 |

Plain truncation beat the document. Restricted to the 20 questions the URL and
title do **not** answer, which are the only ones that test anything:

| context given | agreement |
|---|---|
| SDF document | **0.500** |
| same-size raw page text | **0.900** |

On exactly the questions that need the page, the document loses the answer half
the time while the same number of tokens of raw text keeps it nine times in ten.

## Why

A `brief`, five `key_points`, a topic list and an entity list capture what a page
is about. The questions ask what it *says*: an exact figure, a named mechanism, a
condition, an order of steps. Abstractive summarisation discards precisely that.
This is a property of the artefact's shape, not a defect in the generation.

## What this does and does not show

It does not show that SDF is worthless. The extracted text averaged 1865 tokens,
so the compression under test was about 1.5×, not the 10× or 100× where a
conversion layer would earn its keep. On a long document the comparison could
invert, because truncation would then be discarding most of the page while a
summary still covers all of it.

It does show that at modest compression the current document shape is worse than
doing nothing clever, and that the gap is concentrated in specific detail.

A first run was discarded rather than reported. Every one of its twelve pages came
from a single blog, because the feed reader drained the first feed before reaching
the others, and the url-and-title floor scored 0.861, leaving almost no dynamic
range between floor and ceiling. The numbers above come from the corrected run.

## The fix the result points at

SDF's own schema already has `claims` and `sections` arrays that the shipping
pipeline barely populates. Keeping the page's own sentences preserves what a
summary destroys, and it cannot hallucinate, because every sentence is copied.

Selecting sentences is itself a typed read: each sentence is scored for whether
it states a specific checkable fact, sixteen at a time over one prefill of the
page, and the highest-scoring ones are kept until the token budget matches the
summary's.

That variant was under measurement when work stopped. On the first 8 of 18 pages,
sentence selection matched or beat the abstractive document on all 8 and beat
same-size truncation on 2. That is partial, the sample is small, and it is not a
result. `probe_summary.py` runs the full comparison.

## Extraction, separately

`probe_extract.py` and `sdf_spans.py` test whether typed reading can produce
values rather than categories. It can, by harvesting candidate spans from the page
and letting the model pick a letter, which makes the value a literal substring of
the source and leaves `none` available. The harvester is the ceiling, and its
coverage over 28 live pages is:

| field | candidate list contained the declared value |
|---|---|
| publish date | 0.88 |
| author | 0.75 |
| headline | 0.58 |

Reaching those numbers meant correcting three things, two of which were faults in
the measurement rather than the method:

- the content extractor was in precision mode, which returns the article and
  discards the headline, byline and dateline, so the candidate lists could not
  contain the answer because the text did not;
- the date comparator only matched year-first strings, so a page rendering
  "September 11, 2026" was scored as failing to contain the date its own markup
  declared as 2026-09-11, which alone moved date coverage from 0.00 to 0.79;
- the gold extractor accepted schema.org `@id` references as names, making every
  TechCrunch author a URL that nothing could match.

Headline coverage is the weak one, and the cause is known rather than mysterious:
tag strips and navigation repeat the headline's keywords, so keyword overlap alone
ranks them above the real title. The selection step itself has not been scored
against the harvester's ceiling yet.
