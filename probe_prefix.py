"""Shared-preamble state caching for decide, tested to Codex's preregistration.

SDF records start with the same ~450-token taxonomy on every page. The reader
now holds that preamble's full state (attention KV plus every linear-attention
layer's recurrent state) in a reserved template sequence and copies it per
request, so only the page itself is prefilled.

Three paths over the same held-out SDF pages:

  A  cached       template hit: copy preamble state, prefill the page only
  B  reference    identical split tokens and chunking, preamble recomputed
  C  ordinary     the preamble and page joined into one string, one prefill

Gates (round2_codex.md, stages 2-4):
  A vs B   bit-identical logits. On hybrid models llama.cpp shares the recurrent
           state cell on seq_cp, so this is also the test that advancing a
           working sequence never corrupts the template across many requests.
  A vs C   zero changed answers; max logit difference <= 1e-3 * max(1, |logit|).
           (C tokenizes the joint string, so small differences are allowed.)
  timing   mean speedup of A over C >= 1.25x.

    python probe_prefix.py <model.gguf> [n]
"""
from __future__ import annotations

import sys
import time

import numpy as np

from yantrik_inference import Field, open_model
from probe_sdf import PARENTS, SUBTYPES, TAXONOMY, split, type_path

PREAMBLE = f"{TAXONOMY}\n\nPAGE TO CLASSIFY\n"


def fields():
    fs = [Field("Which of the ten SDF categories does this page belong to?", PARENTS)]
    for p, subs in SUBTYPES.items():
        fs.append(Field(f"Assuming this page is in the {p} category, "
                        f"which specific kind is it?", subs))
    return fs


def page_record(reader, url, title, text, room):
    head = f"URL: {url}\nTitle: {title}\n\nContent:\n"
    fixed = len(reader.tok(head, special=False)) + 8
    ids = reader.tok(text or "", special=False)[:max(0, room - fixed)]
    return head + reader.llm.detokenize(ids).decode("utf-8", "replace")


def main():
    gguf = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    rows, _ = split(n, "test")
    reader, _ = open_model(gguf, decide_ctx=65536, decide_seq=16, with_chat=False)
    fs = fields()
    pre, _ = reader.split_prompt(PREAMBLE, "")
    suffix_max = max(len(reader.suffix_tokens(f)) for f in fs)
    room = reader.per_seq - len(pre) - suffix_max - 16
    recs = [(page_record(reader, u, t, x, room), gold) for u, t, _, _, x, gold in rows]
    print(f"  {len(recs)} held-out SDF pages; preamble {len(pre)} tokens; "
          f"{len(fs)} fields; template sequence {reader.tpl}\n")

    def run(path):
        out, times = [], []
        for rec, _ in recs:
            t0 = time.time()
            if path == "A":
                a = reader.read(rec, fs, preamble=PREAMBLE)
            elif path == "B":
                a = reader.read_split_uncached(PREAMBLE, rec, fs)
            else:
                a = reader.read(PREAMBLE + rec, fs)
            times.append(time.time() - t0)
            rows_ = reader.last_rows
            out.append((a, [r.copy() for r in rows_] if rows_ is not None else None))
        return out, times

    A, tA = run("A")
    hits, misses = reader.cache_hits, reader.cache_misses
    B, tB = run("B")
    reader.last_rows = None
    C, tC = run("C")

    # A vs B: bit-exact
    exact = sum(all(np.array_equal(x, y) for x, y in zip(a[1], b[1])) for a, b in zip(A, B))
    maxdiff_ab = max(max(float(np.abs(x - y).max()) for x, y in zip(a[1], b[1]))
                     for a, b in zip(A, B))
    ans_ab = sum(x.answer != y.answer for a, b in zip(A, B) for x, y in zip(a[0], b[0]))
    # A vs C: answers, and the ordinary path's rows are not stored, compare answers
    ans_ac = sum(x.answer != y.answer for a, c in zip(A, C) for x, y in zip(a[0], c[0]))
    conf_ac = max(abs(x.confidence - y.confidence)
                  for a, c in zip(A, C) for x, y in zip(a[0], c[0]))

    def acc(res, offset=0):
        okp, oks = [], []
        for (a, _), (_, (gp, gs)) in zip(res, recs[offset:]):
            p, s, _, _ = type_path(a)
            okp.append(p == gp)
            if gs:
                oks.append(p == gp and s == gs)
        return np.mean(okp), np.mean(oks)

    th = tA[1:]                                    # hits only
    print(f"  cache: {misses} miss, {hits} hits")
    print(f"  A vs B (cached vs same split from scratch): {exact}/{len(recs)} pages "
          f"bit-identical, max logit diff {maxdiff_ab:.3g}, {ans_ab} answers differ")
    print(f"  A vs C (cached vs ordinary one-string read): {ans_ac} of "
          f"{len(recs) * len(fs)} answers differ, max confidence diff {conf_ac:.4f}")
    print(f"\n  {'path':34s} {'median s':>9s} {'mean s':>7s} {'parent':>7s} {'path':>6s}")
    # accuracy over ALL pages for every path, so the three rows are comparable
    # (an earlier version paired A[1:] with recs[0:] and printed garbage)
    for name, t, res in (("A cached (hits)", th, A), ("B split, recomputed", tB, B),
                         ("C ordinary", tC, C)):
        p, s = acc(res)
        print(f"  {name:34s} {np.median(t):9.3f} {np.mean(t):7.3f} {p:7.3f} {s:6.3f}")
    print(f"\n  speedup of cached over ordinary: mean {np.mean(tC) / np.mean(th):.2f}x, "
          f"median {np.median(tC) / np.median(th):.2f}x, "
          f"p95 {np.percentile(tC, 95) / np.percentile(th, 95):.2f}x   (gate: mean >= 1.25x)")


if __name__ == "__main__":
    main()
