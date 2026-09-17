"""The same 26 real decisions, given the context a real agent harness would have.

probe_context.py showed that what the model is told dominates everything else,
but its richest record was 423 tokens of prose I wrote by hand. A real harness
does not hand-write a summary: it has the repository listing, the machine state,
the tool schemas, and the last several turns verbatim.

This builds the record from actual sources — `git ls-files`, `nvidia-smi`, the
running processes, the real conversation — and scales it up, to answer two
questions the design rests on:

  does accuracy keep improving as the record gets closer to what an agent has?
  does the cost per field stay flat as the record grows?

The second matters most. The record is prefilled once and shared by every field,
so if cost per field stays flat while accuracy rises, then the expensive thing
(context) is bought once and the cheap thing (questions) scales freely.

    python probe_realcontext.py <model.gguf>
"""
from __future__ import annotations

import subprocess
import sys
import time

import numpy as np

from yantrik_inference import Field, open_model
from probe_selfroute import CASES, TOOLS

# What a harness would put in front of the model, in the order it would appear.
TOOL_SCHEMAS = """TOOLS AVAILABLE
  bash    run a shell command on this machine. Use for: checking state, measuring,
          running tests or benchmarks, ssh to other hosts, starting and stopping
          processes, git. Has a timeout and returns stdout and stderr.
  write   create a new file at a path that does not exist yet, with full contents.
  edit    replace an exact string inside a file that already exists. Fails if the
          file has not been read, or if the string is not unique.
  read    open an existing file and return its contents with line numbers.
  search  query the public web. Use for things that happened recently or that are
          outside this machine and this conversation.
  answer  produce a reply directly, calling no tool. Use when the question is about
          something already established, or asks for judgement, explanation or a
          recommendation rather than an action.
  agent   delegate to a sub-agent for a large open-ended task spanning many files
          or sources, where only the conclusion is needed back."""

CONVERSATION = """RECENT CONVERSATION (most recent last)
  user: What about separate pools. Can the context size be different for these
        two different groups. Because for tools maybe 8k is enough whereas for
        chat 32k for example
  assistant: built pools over one set of weights, sized independently; measured
        3.1x throughput from a pool of 2, chat unaffected by concurrent decides
  user: That's odd, can you please check our node4 llm server and see the size?
  assistant: node4 is 192.168.4.154; its LLM server is CT135 llm-vulkan at
        192.168.4.180 on an AMD R9700; the box I had been comparing against was a
        different machine
  user: Something is wrong, please check aig deployment in our k8s
  assistant: the kubeconfig client certificate expired yesterday; the cluster and
        the aig ingress are healthy, verified end to end
  user: Let's forget it. Let's continue with our implementation
  assistant: corrected the sizing guidance in code and README
  user: Let's do some serious testing
  assistant: adversarial probes found a prompt-injection hole and a digit-option
        bug; both fixed and pushed
  user: Can you try to route your own tooling decisions and see how it holds up
  assistant: 0.731 on 26 real decisions against a 0.346 baseline
  user: We might be missing something
  assistant: the record's richness dominates; a bare sentence scores 0.423 and a
        rich one 0.846, and the prefix is amortised across fields"""

SHOTS = """WORKED EXAMPLES
  "check if the service is up"              -> bash    (a command must run)
  "what did we decide about the cache?"     -> answer  (already in the conversation)
  "add a --verbose flag"                    -> edit    (the file exists, change it)
  "build a script that benchmarks this"     -> write   (no such file yet)
  "what is the latest numpy release?"       -> search  (external and current)
  "read the handoff and pick up"            -> read    (open a named file)
  "go through every paper and summarise"    -> agent   (large, open-ended)
  "is that a good idea?"                    -> answer  (judgement, not action)
  "the deploy failed again, look at it"     -> bash    (inspect real state)
  "explain that in simpler terms"           -> answer  (nothing new is needed)"""


def capture(cmd, limit=1200):
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                             timeout=60).stdout.strip()
        return out[:limit]
    except Exception:                                    # noqa: BLE001
        return "(unavailable)"


def real_state():
    repo = capture("git ls-files | head -20")
    commits = capture("git log --oneline -6")
    gpus = capture('nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu '
                   '--format=csv,noheader')
    models = capture("ls ~/.cache/huggingface/hub | grep models-- | head -8")
    return f"""MACHINE AND REPOSITORY STATE
  files in this repository:
{chr(10).join('    ' + l for l in repo.splitlines())}
  recent commits:
{chr(10).join('    ' + l for l in commits.splitlines())}
  GPUs (index, used, total, utilisation):
{chr(10).join('    ' + l for l in gpus.splitlines())}
  model caches present:
{chr(10).join('    ' + l for l in models.splitlines())}
  a server process is holding a 27B model on port 8020; one GPU is idle
  remote hosts reachable over ssh: a Proxmox node, a llama.cpp server"""


def main():
    reader, _ = open_model(sys.argv[1], decide_ctx=32768, decide_seq=8, with_chat=False)
    state = real_state()

    layers = {
        "situation only": lambda said, ctx: (
            f'The user said: "{said}"\nSituation: {ctx}\nAvailable tools: {", ".join(TOOLS)}.'),
        "+ tool schemas": lambda said, ctx: (
            f'{TOOL_SCHEMAS}\n\nThe user said: "{said}"\nSituation: {ctx}'),
        "+ machine state": lambda said, ctx: (
            f'{TOOL_SCHEMAS}\n\n{state}\n\nThe user said: "{said}"\nSituation: {ctx}'),
        "+ conversation": lambda said, ctx: (
            f'{TOOL_SCHEMAS}\n\n{state}\n\n{CONVERSATION}\n\n'
            f'The user said: "{said}"\nSituation: {ctx}'),
        "+ examples (full)": lambda said, ctx: (
            f'{TOOL_SCHEMAS}\n\n{state}\n\n{CONVERSATION}\n\n{SHOTS}\n\n'
            f'The user said: "{said}"\nSituation: {ctx}'),
    }
    q = Field("Which tool should be used first?", TOOLS)

    print(f"  {len(CASES)} real decisions, majority baseline 0.346\n")
    print(f"  {'record':20s} {'tokens':>7s} {'accuracy':>9s} {'≥0.85':>7s} {'acc there':>10s} {'s/call':>7s}")
    best_rec = None
    for name, make in layers.items():
        ok, conf, times, ntok = [], [], [], 0
        for said, ctx, gold in CASES:
            rec = make(said, ctx)
            ntok = len(reader.tok(reader.head + reader.framed(rec), bos=True))
            t0 = time.time()
            a = reader.read(rec, [q])[0]
            times.append(time.time() - t0)
            ok.append(a.answer == gold); conf.append(a.confidence)
        ok, conf = np.array(ok, float), np.array(conf)
        hi = conf >= 0.85
        print(f"  {name:20s} {ntok:7d} {ok.mean():9.3f} {hi.mean()*100:6.0f}% "
              f"{(ok[hi].mean() if hi.any() else float('nan')):10.3f} {np.median(times):7.2f}")
        best_rec = make

    # does the per-field cost stay flat once the record is large?
    said, ctx, _ = CASES[0]
    rec = best_rec(said, ctx)
    ntok = len(reader.tok(reader.head + reader.framed(rec), bos=True))
    extra = [Field(f"Auxiliary question {i}: is this urgent?", ("low", "normal", "high"))
             for i in range(7)]
    print(f"\n  cost of more questions against a {ntok}-token record:")
    print(f"  {'fields':>7s} {'seconds':>8s} {'ms/field':>9s}")
    for n in (1, 2, 4, 8):
        fs = ([q] + extra)[:n]
        reader.read(rec, fs)
        ts = [(lambda: (lambda s: (reader.read(rec, fs), time.time() - s)[1])(time.time()))()
              for _ in range(3)]
        t = float(np.median(ts))
        print(f"  {n:7d} {t:8.3f} {t/n*1000:9.0f}")


if __name__ == "__main__":
    main()
