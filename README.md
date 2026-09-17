# yantrik-inference

Most calls software makes to a language model are not writing. They are deciding.
Is this transaction fraud, which queue does this ticket belong in, is this memory
still relevant, should this escalate. Every one of those has a known set of
answers, and today the usual way to get one is to ask a chat model to write JSON,
then parse it, validate it, and retry when it is malformed.

A decision with a known set of answers does not need the model to write anything.
Prefill the record once, share that prefix across one sequence per question, and
read every answer from a single batched forward pass, restricted to each field's
allowed tokens. The answer is valid by construction because it is never parsed,
and it arrives with a probability you can act on.

The same loaded weights also serve an ordinary OpenAI-compatible chat, because
the parts of an agent that must actually compose something still need to generate.

```
pip install yantrik-inference
yantrik-inference serve --model unsloth/Qwen3.8-27B-GGUF
# open http://localhost:8020
```

## What it costs

Qwen3.8-27B (Q4_K_M, 16.5 GB) on one RTX 3090 Ti, 28 typed questions about one
record, 8-bit KV cache. Reproduce with `yantrik-inference bench`:

| how | accuracy | valid | seconds | vs one pass |
|---|---|---|---|---|
| **all fields, one pass** | **1.000** | 1.00 | **2.03** | 1× |
| one field at a time, sharing the prefix | 1.000 | 1.00 | 2.19 | 1.08× |
| one field at a time, re-reading the record | 1.000 | 1.00 | 6.01 | 2.97× |
| generate the answers as JSON | 1.000 | 1.00 | 10.81 | 5.34× |

Always answering each field's most common value scores 0.655 on this task. That
number is computed from the task generator before any model runs, and it is the
reason the benchmark is trustworthy: an early version of this code used a plain
completion prompt, answered "yes" to every boolean, and scored 0.35 — below the
baseline, which is how the bug was found.

The batched read and the sequential read agree on every field (168/168), so the
mechanism changes how long answers take, not what they are.

**Where the speed comes from, honestly.** Most of it is not generating: 10.81 s to
2.19 s. The batched pass adds a further 1.08×. On a machine with a fast
sequence-copy primitive, the batching is the clean way to share a prefix rather
than the source of the win. On CPU it is worth about 1.1× and nothing more,
because batching converts a memory-bandwidth-bound workload into a compute-bound
one and a CPU is already compute-bound.

**Generation is not accelerated and cannot be.** A reply of N tokens needs N
sequential passes. On the same model and card that is 24 tokens/s with a
one-second first token. The point of this project is to keep decisions out of
that path, not to make that path faster.

## Which models work

Any GGUF that your llama.cpp build can load. Measured here on 12 fields, 4 cases,
with the majority baseline at 0.729 — a model scoring below that is guessing, and
`bench` says so:

| model | accuracy | s/case |
|---|---|---|
| Qwen3.8-27B | 1.000 | 0.98 |
| Qwen2.5-14B | 1.000 | 0.50 |
| Qwen3.5-4B | 1.000 | 0.25 |
| gemma-2-2b-it | 1.000 | 0.17 |
| Qwen3.5-2B | 0.979 | 0.16 |
| Qwen3.5-0.8B | 0.938 | 0.14 |
| Llama-3.2-3B | 0.896 | 0.16 |
| Qwen2.5-0.5B | 0.708 | 0.12 | ← below the baseline; too small for the job |

Three template families (ChatML, Llama 3, Gemma) and two architectures (dense and
hybrid linear-attention) load and answer without any per-model configuration. The
practical floor is about 2B: below that a model tends to collapse onto one answer,
which `bench` reports as a warning rather than a plausible-looking score.

**Run `bench` before you trust a model.** It is four lines of output and it is the
difference between a fast answer and a fast wrong answer.

## Pools, and sizing the two jobs separately

The weights are read-only during inference, so many contexts can share one copy
and run at the same time. Adding a worker costs its cache and compute buffers,
not another copy of the model. The two jobs are sized independently because they
are not alike:

```bash
yantrik-inference serve -m model.gguf \
  --decide-pool 2 --decide-ctx 8192 --decide-seq 16 \
  --chat-pool 1   --chat-ctx 32768
```

Decide workers split their budget across sequences, one per question: 8k over 8
gives 1024 tokens for a record plus a question. Chat wants one long sequence, so
it gets 32k to itself. Keep `--decide-seq` to the number of fields you actually
ask at once — see the sizing note below for why.

Measured on a 27B on one RTX 3090 Ti, three routing questions per call:

| | |
|---|---|
| four calls one at a time | 9.74 s |
| the same four with a pool of 2 | 3.18 s (**3.1× throughput**) |
| a chat alone | 3.99 s |
| the same chat while three decides run | 4.15 s |

Chat is essentially unaffected by concurrent decisions, which is the point: an
agent can route while a conversation is streaming.

`GET /health` reports each pool's size, how many workers are idle, and how often a
request had to queue — that last number is what tells you to raise a pool size.

**Sizing: sequences cost, context is cheap.** On a hybrid model — Qwen3.x and
anything else that mixes linear attention with full attention — most layers keep a
fixed-size recurrent state *per sequence*, and that state does not care how long
the context is. Measured on Qwen3.8-27B, per context, in MiB:

| context | sequences | KV cache | recurrent state | total |
|---|---|---|---|---|
| 8k | 1 | 272 | 150 | 422 |
| 8k | 16 | 272 | 2394 | 2666 |
| 8k | 32 | 272 | 4788 | 5060 |
| 32k | 1 | 1088 | 150 | 1238 |
| 128k | 1 | 4352 | 150 | 4502 |

A **128k** context with one sequence is cheaper than **8k** split 32 ways. So
`--decide-seq` is the expensive knob: set it to the number of fields you really
ask at once, and be generous with `--chat-ctx`, which is nearly free. The server
prints the estimate for your model at startup, and it is derived from the model's
own metadata rather than assumed — it matches llama.cpp's reported buffers within
1.5% on the models tested. Batch size only moves the compute buffer, about 0.5 GB
at 512 against 2.0 GB at 2048, and each pool sizes it automatically.

If a pool does not fit, the server builds a smaller one and says so rather than
failing to start.


## On real decisions, not a generated task

The benchmark above scores 1.000 because the task generator writes clean records
with the facts stated plainly. That is the ceiling of the generator, not of the
method. `probe_selfroute.py` scores 26 real routing decisions taken from the
session that built this repository, labelled with the tool actually used.

What the model is given matters more than anything else measured here:

| the record it sees | tokens | accuracy | ≥0.85 confidence | accuracy there |
|---|---|---|---|---|
| the user's sentence alone | 77 | 0.423 | 50% | 0.538 |
| plus one line of situation | 88 | 0.731 | 58% | 0.867 |
| plus working tree, running jobs, recent turns | 245 | 0.769 | 58% | 0.933 |
| **plus eight worked examples** | 423 | **0.846** | **73%** | **0.947** |

Majority baseline is 0.346. So a starved record halves the accuracy, and the
difference between a usable router and a poor one is mostly what you put in front
of it.

**Examples are nearly free here, which is the structural point.** The record is
prefilled once and every field reads from that same prefix, so state and examples
are paid for once per call however many questions follow:

| questions in one call | seconds | per question |
|---|---|---|
| 1 | 0.55 | 547 ms |
| 2 | 0.63 | 313 ms |
| 4 | 0.78 | 194 ms |
| 8 | 1.12 | 140 ms |

The same eight questions asked as eight separate calls take 4.37 s against 1.12 s
in one — **3.9×** — because each call would otherwise re-read the whole prefix.
In an ordinary setup few-shot examples cost per call; here they cost once and
serve the batch, so the thing that most improves accuracy is also the thing this
design makes cheapest.

**The honest operating point.** With a rich record, 73% of decisions come back
above 0.85 confidence and are right 94.7% of the time; the remaining 27% should
go to a bigger model or a person. Routing everything unconditionally gets 0.846.

One thing tried and rejected: decomposing the seven-way choice into five binary
questions plus a rule scored 0.615 against 0.731 on the same cases, even though
each binary looked good alone. Ask the N-way question and gate on confidence.


### Spending more where it is unsure does not help

The obvious next move is adaptive compute: act on the confident fields, and give
the uncertain ones a second, more expensive pass. Three versions were measured on
the seven low-confidence fields of the routing set, all triggered at 0.85:

| second pass | accuracy on those fields | extra cost |
|---|---|---|
| none | 0.571 | — |
| let the model write a rationale first | **0.429** | 2.0 s per field |
| runoff: binary against each rival | 0.571 | 3.4 s per field |
| self-consistency: 5 permuted votes | 0.571 | 2.9 s per field |

Nothing helped, and generating a rationale actively hurt. The confidence is not
reporting "did not try hard enough"; it is reporting genuine ambiguity in the
question. Looking at the seven cases, four are answered correctly at 50–75%
confidence, and two of the three misses are defensible readings — "is it able to
code?" could reasonably be answered rather than tested.

So the right response to low confidence is **escalation, not more compute**: send
it to a person, a larger model, or go and get more information. That last one is
the only lever that moved anything in these measurements, and it belongs in the
record before the first pass, not after it.

Self-consistency returning identical results is a consistency check passing: the
reader does not change its answer when the options are permuted, so voting over
permutations cannot add information.


## Adversarial testing

The synthetic benchmark measures the happy path. `probe_adversarial.py` attacks
the failure modes that matter when this sits behind an agent. On Qwen3.8-27B:

| probe | result |
|---|---|
| option order — same question, options permuted | stable, 0/3 answers changed |
| position bias — uninformative record | picked the first option 6/12 times |
| **prompt injection through the record** | **blocked by the guard; see below** |
| absence — the fact is not in the record | forced to choose it answers at 0.87 mean confidence; given an `unknown` option it takes it 2/3 times |
| contradiction — the record says both | answers at 100% confidence — it does not notice |
| calibration on genuinely ambiguous cases | 0.925 accuracy, expected calibration error 0.032; the 0.56-confidence bucket is right 50% of the time |
| 10-way enum | correct at 100% |
| batched vs sequential over random cases | 96/96 identical |

**A record is untrusted input.** Text inside it can steer the answer, and two
things are done about it. First, the record and the question are tokenized with
control markers disabled, so `<|im_start|>system` inside a record is literal text
rather than a real turn boundary. Second, the record is wrapped and labelled as
data. Measured on an urgency question whose true answer is `low`:

| attack | plain | delimited only | guarded (default) |
|---|---|---|---|
| fake system turn | **high 86%** | **high 73%** | low 98% |
| appeal to authority | **high 98%** | **high 80%** | low 98% |
| direct instruction | low 98% | low 99% | low 99% |
| urgency claim | low 52% | low 46% | low 99% |

The guard costs about 45 tokens of the shared prefix and no accuracy (still 1.000
on the benchmark). `--no-guard` turns it off.

**Two limits worth knowing.** The model does not notice self-contradiction: given
a record that says both, it answers confidently. And when a fact is absent it
will invent an answer unless you give it an `unknown` option — so give it one for
anything you route on.


## Use it

### From the command line

```bash
# answer questions about a record, once
yantrik-inference decide -m model.gguf \
  --record ticket.txt \
  --ask "Is this a login problem? ; Which team should take it? | billing/identity/platform ; How urgent is it? | low/normal/high"

#   yes       96.3%  Is this a login problem?
#   identity  99.3%  Which team should take it?
#   high      99.4%  How urgent is it?
```

### Over HTTP

```bash
yantrik-inference serve -m model.gguf --port 8020
```

```bash
curl localhost:8020/v1/decide -H 'content-type: application/json' -d '{
  "record": "Ticket 4471: customer cannot log in after the password reset email never arrived.",
  "questions": [
    "Is this a login problem? | yes/no",
    "Which team should take it? | billing/identity/platform",
    "How urgent is it? | low/normal/high"
  ]}'
```

```json
{"seconds": 0.91, "answers": [
  {"question": "Is this a login problem?", "answer": "yes", "confidence": 0.998},
  {"question": "Which team should take it?", "answer": "identity", "confidence": 0.962},
  {"question": "How urgent is it?", "answer": "normal", "confidence": 0.713}]}
```

Chat is OpenAI-compatible on the same port, so existing code only changes its
base URL:

```bash
curl localhost:8020/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Explain a B-tree in one sentence."}],"stream":true}'
```

### From Python

```python
from yantrik_inference import Field, open_model

reader, chat = open_model("model.gguf")

answers = reader.read(record, [
    Field.parse("Is this a login problem? | yes/no"),
    Field.parse("Which team? | billing/identity/platform"),
])
for a in answers:
    if a.confidence < 0.8:
        send_to_human(a)          # the probability is calibrated, so this works
    else:
        act(a.answer)
```

## Confidence you can route on

The number attached to each answer is the softmax over that field's allowed
tokens. On the 27B its expected calibration error is 0.002, and on a 4B it is
0.018, measured over confidence bins. Fields answered at 60% confidence really
are right about 60% of the time, which is what makes a threshold meaningful.

## Installing

You need `llama-cpp-python` built for your accelerator. It is deliberately not a
hard dependency, because a CPU wheel installed by accident is the difference
between 2 seconds and 30.

```bash
# NVIDIA
CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-binary llama-cpp-python llama-cpp-python

# Apple silicon
CMAKE_ARGS="-DGGML_METAL=on" pip install --no-binary llama-cpp-python llama-cpp-python

# CPU only (works, but the whole point is mostly lost)
pip install "yantrik-inference[cpu]"
```

Check it took: `python -c "from llama_cpp import llama_cpp as C; print(C.llama_supports_gpu_offload())"`

Models are any GGUF. `--model` takes a path, or a Hugging Face repo to pull:

```bash
yantrik-inference pull -m unsloth/Qwen3.8-27B-GGUF          # picks a Q4_K_M
yantrik-inference pull -m unsloth/Qwen3.8-27B-GGUF:Q4_K_XL  # or name one
```

## Sizing

llama.cpp divides a context's token budget by its sequence count, and the decide
context needs one sequence per question. So:

```
--decide-ctx 24576 --decide-seq 28   ->  877 tokens for the record plus one question
```

Raise `--decide-seq` for more fields, and `--decide-ctx` in proportion. Latency
grows with the allocated context even when the work is identical: on the 27B the
same ten fields took 1.05 s at a 16k budget and 4.2 s at 32k, so do not allocate
more than you need.

`--kv-type q8_0` is the default. It halves the cache with no measured cost:
perplexity 5.675 against 5.679 at 16-bit on 2500 tokens of prose. It does require
flash attention, which changes the model's top token at about 2% of positions on
near-ties, with no change in perplexity. Use `--kv-type f16` if you need
bit-identical reproducibility more than you need the memory.

## What is and is not new here

The mechanism is not ours. Reading structured decisions from one prefill in
parallel is what [TypeSafe's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
sells, and others have shown it reproduces on stock open models. Jev's advantage
over a stock model is its training, which is undisclosed; on TypeSafe's own
benchmark an independent reproduction with an off-the-shelf 7B reached 73.8%
against Jev's published 86.6%.

What this repository contributes is a clean, measured, open implementation you
can run on your own weights, with the baselines stated and a `bench` command so
you can check the claims on your own hardware rather than believing a table.

## Licence

Apache 2.0.
