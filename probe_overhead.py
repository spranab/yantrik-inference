"""How much of a local read is the GPU, and how much is my own Python?

Before attributing a speed gap to someone else's hardware, account for the parts
of it that are mine. `decode()` fills a llama_batch with a Python loop, four
ctypes assignments per token, so a 1200-token prefill is several thousand
interpreter-level operations before any compute starts.

This times the batch fill separately from llama_decode, and reports prefill
throughput, so the local number can be decomposed instead of hand-waved.

    python probe_overhead.py <model.gguf>
"""
from __future__ import annotations

import ctypes
import sys
import time

import numpy as np

from yantrik_inference import Field, open_model
from yantrik_inference.engine import decode

FILLER = ("The quick brown fox jumps over the lazy dog near the riverbank while "
          "the heron watches from the shallows and the morning light moves. ")


def fill_only(C, tokens):
    """Exactly the work decode() does before calling llama_decode."""
    n = len(tokens)
    batch = C.llama_batch_init(n, 0, 1)
    try:
        t0 = time.perf_counter()
        for i in range(n):
            batch.token[i] = int(tokens[i])
            batch.pos[i] = i
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = 0
            batch.logits[i] = 0
        batch.n_tokens = n
        return time.perf_counter() - t0
    finally:
        C.llama_batch_free(batch)


def main():
    reader, _ = open_model(sys.argv[1], decide_ctx=65536, decide_seq=16,
                           with_chat=False)
    C = reader.C
    fields = [Field(f"Is item {i} urgent?", ("low", "normal", "high"))
              for i in range(11)]

    print(f"  {'record':>8s} {'python fill':>12s} {'full read':>11s} "
          f"{'fill share':>11s} {'prefill tok/s':>14s}")
    for chars in (2000, 5000, 9000):
        record = (FILLER * (chars // len(FILLER) + 1))[:chars]
        prefix = reader.tok_prompt(reader.head, reader.framed(record), bos=True)
        P = len(prefix)

        fills = [fill_only(C, prefix) for _ in range(5)]
        fill = float(np.median(fills))

        reader.read(record, fields)                      # warm
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            reader.read(record, fields)
            ts.append(time.perf_counter() - t0)
        full = float(np.median(ts))

        # prefill alone, without the question batch
        reader.mem.clear()
        t0 = time.perf_counter()
        reader.prefill(prefix)
        pre = time.perf_counter() - t0

        print(f"  {P:8d} {fill * 1000:11.1f}ms {full * 1000:10.0f}ms "
              f"{fill / full * 100:10.1f}% {P / pre:14.0f}")

    print("\n  'python fill' is my own batch construction; the rest is llama.cpp.")
    print("  If the share is small, the local number is GPU-bound and the gap to a")
    print("  hosted service is model size, stack and hardware, not my loop.")


if __name__ == "__main__":
    main()
