"""`yantrik-inference bench` — reproduce the numbers on your own hardware.

Every row is measured on the same cases, and the majority baseline is computed
from the task generator before any model runs, so a broken reader shows up as
scoring below it rather than as a plausible-looking number.
"""
from __future__ import annotations

import json
import random
import time
from typing import List

from .engine import Limits, chat_parts, load_model
from .reader import Field, FieldReader
from .tasks import Case, majority_baseline, make_case


def run(model_path: str, *, cases: int = 10, fields: int = 28, limits: Limits,
        n_batch: int = 2048, n_gpu_layers: int = -1, main_gpu: int = 0,
        split: bool = False, seed: int = 0, with_json: bool = True,
        out: str | None = None) -> dict:
    rng = random.Random(seed)
    raw: List[Case] = [make_case(rng) for _ in range(cases)]
    raw = [(t, qs[:fields]) for t, qs in raw]
    base = majority_baseline(raw)

    t0 = time.time()
    llm, C = load_model(model_path, n_ctx=limits.decide_ctx, n_batch=n_batch,
                        n_seq=limits.decide_seq, n_gpu_layers=n_gpu_layers,
                        main_gpu=main_gpu, split=split, kv_type=limits.kv_type)
    head, tail = chat_parts(llm)
    reader = FieldReader(llm, C, head, tail, limits.decide_seq, limits.per_seq)
    load_s = time.time() - t0

    name = model_path.replace("\\", "/").rsplit("/", 1)[-1]
    print(f"{name}: loaded in {load_s:.0f}s; {cases} cases x {fields} fields, "
          f"{limits.decide_seq} sequences x {limits.per_seq} tokens, cache {limits.kv_type}")
    print(f"  majority baseline (from the generator, before any model ran): {base:.3f}\n")

    methods = [
        ("parallel", lambda r, f: reader.read(r, f)),
        ("one at a time, cached", lambda r, f: reader.read_sequential(r, f, True)),
        ("one at a time, re-read", lambda r, f: reader.read_sequential(r, f, False)),
    ]
    results, preds = {}, {}
    for label, fn in methods:
        acc, times = [], []
        for text, qs in raw:
            fs = [Field(q, o) for q, o, _ in qs]
            t1 = time.time()
            answers = fn(text, fs)
            times.append(time.time() - t1)
            got = [a.answer for a in answers]
            preds.setdefault(label, []).append(got)
            acc.append(sum(g == gold for g, (_, _, gold) in zip(got, qs)) / len(qs))
        results[label] = dict(accuracy=sum(acc) / len(acc),
                              seconds=sum(times) / len(times), valid=1.0)
        print(f"  {label:24s} acc {results[label]['accuracy']:.3f}  valid 1.00  "
              f"{results[label]['seconds']:6.2f}s/case", flush=True)

    if with_json:
        # Generation needs one long sequence, which the decide context cannot
        # give (its budget is divided by its sequence count). Build a second
        # context over the same weights, as the server does for chat.
        from .chat import ChatEngine
        longest = max(len(reader.tok(reader.head + t, bos=True)) for t, _ in raw)
        gen_ctx = 1 << max(12, (longest + 40 * fields + 900).bit_length())
        gen = ChatEngine(llm, C, gen_ctx, n_batch, kv_type=limits.kv_type)
        print(f"  (the JSON row generates, so it runs on a second {gen_ctx}-token context)")

        def generate(body: str, max_new: int) -> str:
            return "".join(gen.stream([dict(role="user", content=body)],
                                      max_tokens=max_new, temperature=0.0))

        acc, times, valid = [], [], []
        for text, qs in raw:
            fs = [Field(q, o) for q, o, _ in qs]
            t1 = time.time()
            vals, _ = reader.read_as_json(text, fs, max_new=900, generate=generate)
            times.append(time.time() - t1)
            valid.append(vals is not None and all(v is not None for v in vals))
            vals = vals or [None] * len(fs)
            acc.append(sum(v == gold for v, (_, _, gold) in zip(vals, qs)) / len(qs))
        if times:
            results["generate as JSON"] = dict(
                accuracy=sum(acc) / len(acc), seconds=sum(times) / len(times),
                valid=sum(valid) / len(valid))
            r = results["generate as JSON"]
            print(f"  {'generate as JSON':24s} acc {r['accuracy']:.3f}  "
                  f"valid {r['valid']:.2f}  {r['seconds']:6.2f}s/case")

    # agreement between the batched read and the sequential one: the mechanism
    # should not change answers, only how long they take
    a, b = preds.get("parallel"), preds.get("one at a time, cached")
    agree = None
    if a and b:
        tot = sum(len(x) for x in a)
        same = sum(p == q for xa, xb in zip(a, b) for p, q in zip(xa, xb))
        agree = same / tot
        print(f"\n  parallel vs sequential agreement: {agree:.4f} ({same}/{tot} fields)")

    p = results["parallel"]["seconds"]
    print("\n  cost relative to reading all fields in one pass:")
    for label, r in results.items():
        if label != "parallel":
            print(f"    {label:24s} {r['seconds'] / p:5.2f}x")

    payload = dict(model=name, cases=cases, fields=fields, majority_baseline=base,
                   load_seconds=load_s, agreement=agree,
                   limits=dict(decide_ctx=limits.decide_ctx, decide_seq=limits.decide_seq,
                               per_seq=limits.per_seq, kv=limits.kv_type),
                   results=results)
    if out:
        with open(out, "w") as fh:
            json.dump(payload, fh, indent=1)
        print(f"\n  wrote {out}")
    return payload
