"""Ternary Bonsai 2 27B against our Q4_K_M, on our task, through one code path.

PrismML's Bonsai 2 27B is a ternary copy of Qwen3.8-27B, the model this
repository runs, shrunk from 54 GB to 5.9 GB with a claimed 98.2% of benchmark
quality. Their benchmarks are generative and thinking-mode. Ours is typed reading:
the answer is read from one position's logits, with no chance for reasoning to
paper over a damaged distribution. So the question is whether ternary weights
keep the logits honest, not just the prose.

Both models run in PrismML's llama.cpp fork (the only runtime with the Hadamard
activation transform Bonsai needs), behind the same llama-server, scored by the
same code. The reader here reproduces yantrik_inference.reader exactly: same chat
head and tail, same untrusted-record guard, same question suffix, each option
scored as its best first-token spelling, softmax over the options. The only
difference is where the logits come from: the server's top log-probabilities,
wide enough that every option's spellings are present.

Scored against the same URL-derived gold as probe_sdf.py, on the same held-out
pages, with the same eleven questions: the parent category plus one speculative
subtype question per category.

    python probe_bonsai.py <port> <label> [n]
"""
from __future__ import annotations

import json
import math
import sys
import time
import urllib.request
from collections import Counter

import numpy as np

from yantrik_inference.reader import GUARD, Field
from probe_sdf import PARENTS, SUBTYPES, TAXONOMY, split

HEAD = "<|im_start|>user\n"
TAIL = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
TOP_K = 300


class ServerReader:
    def __init__(self, port):
        self.base = f"http://127.0.0.1:{port}"
        self._first: dict = {}
        self.missing = 0

    def post(self, path, body, timeout=600):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def tok(self, s):
        return [t["id"] if isinstance(t, dict) else t
                for t in self.post("/tokenize", {"content": s,
                                                 "with_pieces": True})["tokens"]]

    def pieces(self, s):
        return self.post("/tokenize", {"content": s, "with_pieces": True})["tokens"]

    def detok(self, ids):
        return self.post("/detokenize", {"tokens": ids})["content"]

    def first_tokens(self, option):
        """Every plausible first token, dropping bare whitespace. Same rule as
        FieldReader.first_tokens, so the two paths score options identically."""
        if option not in self._first:
            seen, out = set(), []
            for cand in (option, " " + option, option.capitalize(),
                         " " + option.capitalize(), option.upper()):
                ps = self.pieces(cand)
                if not ps:
                    continue
                first = ps[0]
                if first["id"] in seen or not str(first.get("piece", "")).strip():
                    continue
                seen.add(first["id"]); out.append(first["id"])
            self._first[option] = out
        return self._first[option]

    def read(self, record, fields):
        prefix = HEAD + GUARD.format(record=record)
        out = []
        for f in fields:
            suffix = (f"\n\nQuestion: {f.question}\nAnswer with exactly one word "
                      f"from: {', '.join(f.options)}")
            d = self.post("/completion", {
                "prompt": prefix + suffix + TAIL, "n_predict": 1,
                "n_probs": TOP_K, "temperature": 0, "cache_prompt": True})
            top = d["completion_probabilities"][0]["top_logprobs"]
            lp = {t["id"]: t["logprob"] for t in top}
            v = np.array([max((lp.get(i, -math.inf) for i in self.first_tokens(o)),
                              default=-math.inf) for o in f.options])
            if not np.isfinite(v).any():
                self.missing += 1
                v = np.zeros(len(f.options))
            v = np.where(np.isfinite(v), v, v[np.isfinite(v)].min() - 30
                         if np.isfinite(v).any() else 0)
            e = np.exp(v - v.max()); p = e / e.sum()
            j = int(p.argmax())
            out.append((f.options[j], float(p[j])))
        return out


def fields():
    fs = [Field("Which of the ten SDF categories does this page belong to?", PARENTS)]
    for p, subs in SUBTYPES.items():
        fs.append(Field(f"Assuming this page is in the {p} category, "
                        f"which specific kind is it?", subs))
    return fs


def record_for(rd, url, title, text, budget=4040):
    """Same token budget as probe_sdf.py, counted with this model's tokenizer."""
    head = (f"{TAXONOMY}\n\nPAGE TO CLASSIFY\nURL: {url}\nTitle: {title}\n\n"
            f"Content:\n")
    fixed = len(rd.tok(HEAD + GUARD.format(record=head)))
    ids = rd.tok(text or "")[:max(0, budget - fixed)]
    return head + rd.detok(ids)


def main():
    port, label = int(sys.argv[1]), sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    rows, total = split(n, "test")
    rd = ServerReader(port)
    fs = fields()
    okp, oks, cp, times = [], [], [], []
    errs = Counter()
    for url, title, _, _, text, (gp, gs) in rows:
        rec = record_for(rd, url, title, text)
        t0 = time.time()
        a = rd.read(rec, fs)
        times.append(time.time() - t0)
        parent, pc = a[0]
        sub, _ = a[1 + PARENTS.index(parent)]
        okp.append(parent == gp); cp.append(pc)
        if gs:
            oks.append(parent == gp and sub == gs)
        if parent != gp:
            errs[(gp, parent)] += 1
    okp, oks, cp = np.array(okp, float), np.array(oks, float), np.array(cp)
    hi = cp >= 0.97
    res = {"label": label, "n": len(okp), "parent": okp.mean(), "path": oks.mean(),
           "n_path": len(oks), "median_s": float(np.median(times)),
           "gate_share": float(hi.mean()),
           "gate_acc": float(okp[hi].mean()) if hi.any() else float("nan"),
           "missing": rd.missing, "errors": errs.most_common(5)}
    print(json.dumps(res, default=str))
    print(f"\n  {label}: parent {okp.mean():.3f}  path {oks.mean():.3f} "
          f"(n={len(okp)}/{len(oks)})  median {np.median(times):.2f}s/page  "
          f">=0.97 on {hi.mean()*100:.0f}% at {res['gate_acc']:.3f}  "
          f"options-missing {rd.missing}")
    for (g, p), c in errs.most_common(5):
        print(f"    {g:16s} -> {p:16s} {c}")


if __name__ == "__main__":
    main()
