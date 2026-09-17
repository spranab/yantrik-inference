"""Route this session's own tooling decisions and score them against what was done.

The synthetic benchmark has an oracle but no external validity: the records are
generated, so the task can only be as hard as the generator. These cases are
real. Each is something the user actually said during the session that built this
repository, and the label is the tool that was actually reached for first.

That makes it an honest test of the agentic use case, and it is deliberately not
cherry-picked: every substantive user turn is here, including the ones where the
right move was to answer rather than act, which is the decision agents get wrong
most often.

Tools, as an agent would see them:
    bash    run a command: measure, check, ssh, launch, kill
    write   create a new file
    edit    change an existing file
    read    open a file to learn what is in it
    search  look something up on the web
    answer  no tool; reply from what is already known
    agent   delegate to a sub-agent

    python probe_selfroute.py <model.gguf>
"""
from __future__ import annotations

import sys
import time

import numpy as np

from yantrik_inference import Field, open_model

TOOLS = ("bash", "write", "edit", "read", "search", "answer", "agent")

# (what the user said, a line of situation, the tool actually used)
CASES = [
    ("Hey bro, lets pickup in this system",
     "Start of session. A handoff document exists in the repository.", "read"),
    ("Now the question again, is there any future for this?",
     "The results are already in context; nothing new has happened.", "answer"),
    ("Can you please check with codex why our model did not learn",
     "Codex is a CLI installed on this machine.", "bash"),
    ("Ok, soo. what next",
     "All findings are in context; the user is asking for a recommendation.", "answer"),
    ("lauch one more agent: go through different sculptures of Hinduism",
     "A long open-ended research task over many sources.", "agent"),
    ("is it able to code?",
     "A trained model and an evaluation script already exist in the repo.", "bash"),
    ("how is the training going",
     "A training run is in progress writing to a log file.", "bash"),
    ("can it learn unsupervised from wiki?",
     "A design question about what the architecture could do.", "answer"),
    ("can you please stop it for now",
     "A training process is running in the background.", "bash"),
    ("What is the Jev AI from typesafe everyone is talking about",
     "Something announced publicly in the last two days; not in context.", "search"),
    ("Go ahead please",
     "The user approved building a reproduction; no file exists yet.", "write"),
    ("so I can't chat like I do here? or use for agentic stuff?",
     "A capability question about what was just built.", "answer"),
    ("the speed is incredible, but for chat will it be the same?",
     "An empirical claim that has not been measured yet.", "bash"),
    ("what what is jev doing? it will not be able to do coding and stuff?",
     "The facts were gathered a few turns ago and are in context.", "answer"),
    ("can we have dual mode? one endpoint generation another jev like",
     "A new server file has to be created.", "write"),
    ("why not got for 8bit for context",
     "An existing loader file needs a new option.", "edit"),
    ("Any issue with flash attention?",
     "A correctness question that needs a sensitive measurement.", "bash"),
    ("This will work for most of the models?",
     "Several models are on disk and the benchmark command exists.", "bash"),
    ("That's odd, can you please check our node4 llm server and see the size?",
     "A remote host reachable over ssh.", "bash"),
    ("Something is wrong, please check aig deployment in our k8s",
     "A remote cluster; kubectl and ssh are available.", "bash"),
    ("Let's forget it. Let's continue with our implementation",
     "Earlier guidance in the code is now known to be wrong.", "edit"),
    ("Let's do some serious testing",
     "No adversarial test file exists yet.", "write"),
    ("So the idea is to run a single server and expose two endpoints",
     "The user is restating the design to confirm understanding.", "answer"),
    ("What about separate pools. Can the context size be different",
     "A new pooling module has to be created.", "write"),
    ("so explain to me in layman terms",
     "Everything needed is already in context.", "answer"),
    ("is thre any unsolved math which could improve learning speed?",
     "A broad question about published work.", "answer"),
]


def main():
    gguf = sys.argv[1]
    reader, _ = open_model(gguf, decide_ctx=16384, decide_seq=8, with_chat=False)

    tool_q = Field("Which tool should be used first?", TOOLS)
    extra = [
        Field("Does this require running a command?", ("yes", "no")),
        Field("Does this create or change a file?", ("yes", "no")),
        Field("Can this be answered from what is already known?", ("yes", "no")),
        Field("Is the user asking a question rather than giving an instruction?", ("yes", "no")),
    ]

    print(f"  {len(CASES)} real decisions from this session, {len(TOOLS)} tools\n")
    rows, times = [], []
    for said, ctx, gold in CASES:
        record = (f"The user said: \"{said}\"\n"
                  f"Situation: {ctx}\n"
                  f"Available tools: {', '.join(TOOLS)}.")
        t0 = time.time()
        a = reader.read(record, [tool_q] + extra)
        times.append(time.time() - t0)
        pred, conf = a[0].answer, a[0].confidence
        rows.append((pred, gold, conf, [x.answer for x in a[1:]]))
        mark = "ok " if pred == gold else "BAD"
        print(f"  {mark} {pred:7s}({conf*100:3.0f}%) want {gold:7s}  {said[:44]}")

    acc = np.mean([p == g for p, g, _, _ in rows])
    conf = np.array([c for _, _, c, _ in rows])
    ok = np.array([p == g for p, g, _, _ in rows], float)
    print(f"\n  accuracy {acc:.3f} over {len(rows)} cases, {np.median(times)*1000:.0f} ms median "
          f"for 5 fields")

    from collections import Counter
    gold_counts = Counter(g for _, g, _, _ in rows)
    top = gold_counts.most_common(1)[0]
    print(f"  majority baseline (always say {top[0]!r}): {top[1]/len(rows):.3f}")

    print("\n  calibration:")
    for lo, hi in ((0, .6), (.6, .85), (.85, .97), (.97, 1.01)):
        m = (conf >= lo) & (conf < hi)
        if m.sum():
            print(f"    conf {lo:.2f}-{hi:.2f}: n={int(m.sum()):2d}  mean {conf[m].mean():.2f}  "
                  f"accuracy {ok[m].mean():.2f}")

    print("\n  where it went wrong:")
    for (pred, gold, c, _), (said, _, _) in zip(rows, CASES):
        if pred != gold:
            print(f"    said {gold!r}, chose {pred!r} at {c*100:.0f}%  <- {said[:52]}")

    # the secondary fields are the ones an agent would actually branch on
    print("\n  secondary fields, agreement with the tool label:")
    needs_cmd = np.mean([(sec[0] == "yes") == (g == "bash") for _, g, _, sec in rows])
    touches = np.mean([(sec[1] == "yes") == (g in ("write", "edit")) for _, g, _, sec in rows])
    known = np.mean([(sec[2] == "yes") == (g == "answer") for _, g, _, sec in rows])
    print(f"    'requires a command'  matches the bash label      {needs_cmd:.2f}")
    print(f"    'creates/changes a file' matches write|edit       {touches:.2f}")
    print(f"    'answerable from known' matches answer            {known:.2f}")


if __name__ == "__main__":
    main()
