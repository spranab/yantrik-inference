"""Does the record's richness explain the gap between 0.73 and the benchmark's 1.00?

probe_selfroute.py gave the model one line of user text and one line of
situation, then reported 0.731 as the method's accuracy on real routing. That
conflates two things: how well the reader works, and how little it was told.

A real agent router has the conversation, the working tree, and what is running.
And this design has a property worth exploiting: the record is prefilled ONCE and
shared by every field, so anything put in it — examples included — is paid for
once and serves all N decisions. In an ordinary setup few-shot examples cost per
call; here they are amortised.

Four records over the same 26 labelled decisions:

  bare       the user's sentence alone
  situation  plus one line of context            (what probe_selfroute used)
  state      plus the working tree, running jobs and the last few turns
  state+shot plus eight worked examples in the shared prefix

    python probe_context.py <model.gguf>
"""
from __future__ import annotations

import sys
import time

import numpy as np

from yantrik_inference import Field, open_model
from probe_selfroute import CASES, TOOLS

STATE = """Working tree: yantrik_inference/{engine,reader,chat,server,pool,cli,bench}.py,
tests/test_reader.py, README.md, probe_adversarial.py. A git repo with a remote.
Running now: a server process on port 8020 holding a 27B model; two GPUs, one idle.
On disk: eight GGUF models in the Hugging Face cache; a CUDA build of llama.cpp in
a venv. Reachable: a Proxmox host over ssh, a llama.cpp server on another box.
Recent turns: the user asked for pools, then for the context sizes to differ, then
for adversarial testing. Everything discussed so far is still in the conversation."""

SHOTS = """Worked examples of the same decision:
- "check if the service is up" -> bash, because it needs a command run.
- "what did we decide about the cache?" -> answer, because it is in the conversation.
- "add a --verbose flag" -> edit, because the file exists and needs a change.
- "build a script that benchmarks this" -> write, because no such file exists.
- "what is the latest version of numpy?" -> search, because it is external and current.
- "read the handoff and pick up" -> read, because a named file must be opened.
- "go through every paper on this and summarise" -> agent, because it is large and open-ended.
- "is that a good idea?" -> answer, because it asks for judgement, not action."""


def build(kind, said, ctx):
    tools = f"Available tools: {', '.join(TOOLS)}."
    if kind == "bare":
        return f'The user said: "{said}"\n{tools}'
    if kind == "situation":
        return f'The user said: "{said}"\nSituation: {ctx}\n{tools}'
    if kind == "state":
        return f"{STATE}\n\nThe user said: \"{said}\"\nSituation: {ctx}\n{tools}"
    return f"{STATE}\n\n{SHOTS}\n\nThe user said: \"{said}\"\nSituation: {ctx}\n{tools}"


def main():
    reader, _ = open_model(sys.argv[1], decide_ctx=32768, decide_seq=8, with_chat=False)
    q = Field("Which tool should be used first?", TOOLS)
    print(f"  {len(CASES)} real decisions; majority baseline 0.346\n")
    print(f"  {'record':12s} {'tokens':>7s} {'accuracy':>9s} {'≥0.85 conf':>11s} {'acc there':>10s} {'ms':>6s}")
    results = {}
    for kind in ("bare", "situation", "state", "state+shot"):
        ok, conf, times = [], [], []
        ntok = 0
        for said, ctx, gold in CASES:
            rec = build(kind, said, ctx)
            ntok = len(reader.tok(reader.head + reader.framed(rec), bos=True))
            t0 = time.time()
            a = reader.read(rec, [q])[0]
            times.append(time.time() - t0)
            ok.append(a.answer == gold); conf.append(a.confidence)
        ok, conf = np.array(ok, float), np.array(conf)
        hi = conf >= 0.85
        results[kind] = (ok.mean(), hi.mean(), ok[hi].mean() if hi.any() else float("nan"))
        print(f"  {kind:12s} {ntok:7d} {ok.mean():9.3f} {hi.mean()*100:10.0f}% "
              f"{ok[hi].mean() if hi.any() else float('nan'):10.3f} {np.median(times)*1000:6.0f}")

    best = max(results, key=lambda k: results[k][0])
    print(f"\n  best record: {best} at {results[best][0]:.3f} "
          f"(probe_selfroute reported {results['situation'][0]:.3f} using 'situation')")
    print("  the shared prefix is prefilled once per call and serves every field,")
    print("  so the examples and state cost one prefill however many questions follow.")


if __name__ == "__main__":
    main()
