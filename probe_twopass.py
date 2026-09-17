"""Spend more only where the reader is unsure.

The single pass answers every field from one forward pass and reports a
calibrated probability. That probability is useful for more than routing to a
human: it says which fields deserve a second, more expensive look.

Second pass, for fields below a threshold only: let the model write a short
rationale, then read the answer from the logits again with the rationale in
context. Generation is the expensive thing this project exists to avoid, but
paying for it on 27% of fields is a different proposition from paying on all.

The question this settles is whether low-confidence fields are fixable or simply
ambiguous. If thinking does not move them, the confidence was reporting real
uncertainty in the task and the right move is escalation, not more compute.

    python probe_twopass.py <model.gguf>
"""
from __future__ import annotations

import sys
import time

import numpy as np

from yantrik_inference import Field, open_model
from yantrik_inference.engine import decode
from probe_context import build
from probe_selfroute import CASES, TOOLS


def rationale_then_answer(reader, chat, record, field, max_think=64):
    """Generate a brief rationale, then read the answer with it in context."""
    ask = (f"{reader.framed(record)}\n\nQuestion: {field.question}\n"
           f"Options: {', '.join(field.options)}.\n"
           f"In one short sentence, say which option the facts support and why.")
    think = "".join(chat.stream([dict(role="user", content=ask)],
                                max_tokens=max_think, temperature=0.0)).strip()
    rec2 = f"{record}\n\nAn analyst notes: {think}"
    return reader.read(rec2, [field])[0], think


def main():
    reader, chat = open_model(sys.argv[1], decide_ctx=32768, decide_seq=8,
                              chat_ctx=8192)
    q = Field("Which tool should be used first?", TOOLS)

    first, times = [], []
    for said, ctx, gold in CASES:
        rec = build("state+shot", said, ctx)
        t0 = time.time()
        a = reader.read(rec, [q])[0]
        times.append(time.time() - t0)
        first.append((rec, a, gold))

    conf = np.array([a.confidence for _, a, _ in first])
    ok1 = np.array([a.answer == g for _, a, g in first], float)
    print(f"  one pass: accuracy {ok1.mean():.3f}, median {np.median(times)*1000:.0f} ms\n")

    for thresh in (0.85, 0.95):
        low = conf < thresh
        print(f"  === second pass for confidence < {thresh} ({int(low.sum())}/{len(CASES)} fields)")
        ok2 = ok1.copy()
        changed = fixed = broke = 0
        t0 = time.time()
        for i, ((rec, a, gold), is_low) in enumerate(zip(first, low)):
            if not is_low:
                continue
            b, why = rationale_then_answer(reader, chat, rec, q)
            ok2[i] = float(b.answer == gold)
            if b.answer != a.answer:
                changed += 1
                fixed += int(b.answer == gold and a.answer != gold)
                broke += int(b.answer != gold and a.answer == gold)
        extra = time.time() - t0
        base_low = ok1[low].mean() if low.any() else float("nan")
        new_low = ok2[low].mean() if low.any() else float("nan")
        print(f"    on those fields: {base_low:.3f} -> {new_low:.3f}"
              f"   ({changed} answers changed: {fixed} fixed, {broke} broken)")
        print(f"    overall:         {ok1.mean():.3f} -> {ok2.mean():.3f}"
              f"   extra time {extra:.1f}s for {int(low.sum())} fields "
              f"({extra/max(1,low.sum()):.2f}s each)")


if __name__ == "__main__":
    main()
