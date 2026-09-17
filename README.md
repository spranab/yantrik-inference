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

## Use it

### From the command line

```bash
# answer questions about a record, once
yantrik-inference decide -m model.gguf \
  --record claim.txt \
  --ask "Is the amount over 10000? ; Which region? | domestic/offshore/regional"

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
