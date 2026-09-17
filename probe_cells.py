"""Is the per-sequence budget the real limit, or is the shared prefix shared?

FieldReader refuses a read when record + question exceeds n_ctx / n_seq, which is
llama.cpp's convention for INDEPENDENT sequences. But this design does not use
independent sequences: llama_memory_seq_cp tags the prefix cells with another
sequence id rather than copying them, so N sequences over one record should
occupy P + sum(len(suffix)) cells in total, not N * (P + len(suffix)).

If that is right, the current check wastes most of the context on exactly the
workload the library exists for, and a page-sized record is refused for no
physical reason. If it is wrong, llama_decode will fail and the check is correct.

Deliberately run with a record far past n_ctx / n_seq and see which happens.

    python probe_cells.py <model.gguf>
"""
from __future__ import annotations

import sys
import time

from yantrik_inference import Field, open_model
from yantrik_inference.engine import decode

FILLER = ("The quick brown fox jumps over the lazy dog near the riverbank at "
          "dawn while the heron watches from the shallows. ")


def main():
    n_ctx, n_seq = 65536, 16
    reader, _ = open_model(sys.argv[1], decide_ctx=n_ctx, decide_seq=n_seq,
                           with_chat=False)
    per_seq = n_ctx // n_seq
    print(f"  n_ctx {n_ctx}, {n_seq} sequences, so the current check allows "
          f"{per_seq} tokens per read\n")

    fields = [Field(f"Is statement {i} about an animal?", ("yes", "no"))
              for i in range(11)]

    for mult in (1, 2, 4, 8):
        record = FILLER * (per_seq * mult // 22)
        prefix = reader.tok_prompt(reader.head, reader.framed(record), bos=True)
        P = len(prefix)
        suffixes = [reader.suffix_tokens(f) for f in fields]
        cells = P + sum(len(s) for s in suffixes)
        naive = len(fields) * (P + max(len(s) for s in suffixes))
        try:
            reader.mem.clear()
            t0 = time.time()
            reader.prefill(prefix)
            for i in range(1, len(suffixes)):
                reader.mem.seq_cp(0, i)
            got = reader.ask(P, suffixes)
            dt = time.time() - t0
            out = reader._pick(got, fields)
            status = f"OK  {dt:5.2f}s  first answer {out[0].answer!r}"
        except Exception as e:                            # noqa: BLE001
            status = f"FAILED  {type(e).__name__}: {str(e)[:60]}"
        print(f"  record {P:6d} tokens ({P/per_seq:4.1f}x the per-seq budget)  "
              f"shared cells {cells:6d}  if not shared {naive:7d}   {status}")

    print("\n  If the large records pass, the prefix is genuinely shared and the")
    print("  limit is total cells against n_ctx, not record size against n_ctx/n_seq.")


if __name__ == "__main__":
    main()
