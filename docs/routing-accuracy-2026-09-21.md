# Can routing accuracy be improved?

2026-09-21. Qwen3.8-27B (UD-Q4_K_M) on one RTX 3090 Ti, through `/v1/decide` in
the llama.cpp fork. One typed question per request, answer read as the argmax
over the option first tokens, shared preamble prefilled once and cached.

The endpoint's own test scored 9 of 10 on a seven-tool router, and the question
was whether that can be made better. Ten cases cannot answer it: one decision is
worth ten points, so 9/10 and 10/10 are the same measurement. This is what two
larger benchmarks say instead.

## The mechanism is not the limit

The first benchmark is a 24-way router built from the 60k labelled calls in
xlam-function-calling-60k: the 24 most frequent tools that have one consistent
spec across the dataset and do not overlap each other, 40 held-out queries each,
balanced, so chance is 0.042. Tool names and one-line descriptions go in the
cached preamble and the query is the record.

It is saturated. 576 of 576 on the split reserved for fitting, and every arm
from bare names upward sits at or above 0.958 on samples of the eval split. When
tools do different jobs and each has a description, a typed read over their
names is already as good as the labels.

So the interesting question is the case the fork's test actually failed on:
`read` chosen for "add a --verbose flag to the CLI parser" where `edit` was
wanted. Both tools touch a file. Nothing in the request names the operation.

## A benchmark of tools that overlap

The second benchmark keeps only confusable options. Tools with a consistent spec
and at least 55 cases are clustered by name-and-description overlap, and each
cluster becomes its own routing problem whose options are its members and
nothing else: `is_perfect_square | is_power | is_power_of_two`,
`calculate_distance | euclidean_distance`, `generate_password |
generate_random_string`, and ten more. 13 clusters, 930 held-out eval cases, 248
shown as examples, 248 reserved for calibration, all disjoint.

Options are asked as letters, because 8 of the 13 clusters contain two tool names
that share a first token and the endpoint refuses those rather than alias them.

Each arm changes only the cached preamble. The cases, the question and the model
are identical, so the arms are paired and the test is McNemar's.

| preamble | accuracy | fixed | broke | p |
|---|---|---|---|---|
| tool names only | 0.870 ±0.022 | | | |
| + one-line descriptions | **0.901** ±0.019 | 31 | 2 | <1e-4 |
| + argument names | 0.902 ±0.019 | 1 | 0 | 1.00 |
| + 8 worked examples per tool | 0.914 ±0.018 | 55 | 44 | 0.31 |
| + boundary rules the model wrote itself | 0.909 ±0.019 | 14 | 19 | 0.49 |
| + per-option calibration fit on held-out cases | 0.916 ±0.018 | 19 | 12 | 0.28 |

n = 930, ± is a 95% interval, each row's test is against the row above it.

One lever survives. Writing a sentence about each tool is worth about three
points and it is not close to chance. After that the ladder stops: examples
churn 55 cases in and 44 out, the model's own disambiguation rules break more
than they fix, and correcting the option bias on held-out cases is within noise.
Median latency is 146-150 ms across every arm, because the preamble is prefilled
once and copied (cached on 917 of 930 requests), so the richer arms are free —
they are simply not better.

Two of these were things I expected to work. The same 27B was asked to write,
from the specs alone, one line per tool saying when to choose it instead of its
neighbours; it did that well for the pairs that differ (`calculate_distance`:
"use when the points have more than two dimensions") and produced nonsense for
the pairs that do not, including two rules that contradicted their own tool
names. Worked examples helped a great deal in an earlier record-reading task, but
there they added context about the record; here the record is one sentence and
the ambiguity is between the options, which examples do not resolve.

## The ceiling is the tool catalogue

Per-cluster accuracy on the best arm:

| accuracy | options |
|---|---|
| 1.000 | `geocode_city \| get_city_from_zipcode \| get_ip_location \| get_ip_zipcode` |
| 1.000 | `circle_area \| triangle_area` |
| 1.000 | `project_investment_growth \| project_population` |
| 1.000 | `dice_roll_probability \| probability_of_consecutive_rolls` |
| 1.000 | `greatest_common_divisor \| least_common_multiple` |
| 1.000 | `find_longest_palindromic_substring \| find_longest_word` |
| 1.000 | `merge_dictionaries \| merge_sorted_lists` |
| 0.978 | `is_perfect_square \| is_power \| is_power_of_two` |
| 0.967 | `generate_password \| generate_random_string` |
| 0.867 | `calculate_distance \| euclidean_distance` |
| 0.792 | `calculate_age \| calculate_median \| calculate_standard_deviation \| std_deviation` |
| 0.767 | `is_anagram \| is_anagram_phrase` |
| 0.550 | `calculate_factorial \| factorial` |

Seven of thirteen clusters are perfect. The error is concentrated in the two
clusters that contain two tools with the *same description* — `std_deviation`
and `calculate_standard_deviation` are documented identically, `factorial` and
`calculate_factorial` differ only in the verb — and in two more whose
distinction is real but absent from the request (`is_anagram` takes words,
`is_anagram_phrase` takes phrases; `euclidean_distance` is 2-D,
`calculate_distance` is n-D). On the eleven clusters that do not contain a
duplicate, accuracy is **0.965** on 750 cases.

No preamble fixes a duplicate. `factorial` against `calculate_factorial` is a
coin flip at 0.550 because the label is arbitrary, and the fix is to delete one
of them. This is worth checking before blaming a router: any pair of tools whose
specs do not distinguish them puts a hard ceiling on routing, and the pairs can
be found offline from the catalogue.

## Letters cost nothing

Because the engine scores an option by its first token, options that share one
are refused, and a caller with names like `merge_dictionaries` and
`merge_sorted_lists` has to label the choices instead. On the 5 clusters of 13
whose names *are* distinguishable, the same 300 cases were asked both ways:

| options | accuracy | mean confidence |
|---|---|---|
| letters | 0.880 ±0.037 | 0.944 |
| tool names | 0.873 ±0.038 | 0.962 |

Names fixed 0 cases and broke 2, p = 0.50. The indirection through a letter is
not costing anything, which also means that teaching the engine to score whole
option strings would buy convenience, not accuracy.

## What to do about low confidence: escalate

Gating on the reported confidence, on the best arm:

| gate | share kept | accuracy kept | accuracy of the rest |
|---|---|---|---|
| 0.50 | 0.97 | 0.927 | 0.625 |
| 0.70 | 0.91 | 0.947 | 0.625 |
| 0.90 | 0.82 | 0.974 | 0.646 |
| 0.97 | 0.78 | 0.989 | 0.662 |
| 0.99 | 0.73 | 0.996 | 0.704 |

The confidence is informative: at 0.97, 78% of decisions are right 98.9% of the
time and the 22% held back would have been right 66% of the time. An agent that
routes the confident ones typed and sends the rest to the model gets most of the
speed and almost none of the errors. This matches the earlier finding on typed
page reading, where second passes — rationales, runoffs, self-consistency — did
nothing, and the useful response to low confidence was escalation rather than
more compute.

## The ten-case failure was the question

With the benchmarks in hand, the original failure is explainable. The question
asked *"Which tool should be used first?"*, and for "add a --verbose flag to the
CLI parser" the model answered `read` at 87%. That is not a mistake: an agent
does read a file before editing it. The label wanted the tool that carries out
the request. Changing the question to match, and then sharpening the tool lines
to say what each one should be chosen *instead of*:

| request | want | "used first" | "carry out" | + sharp spec | + examples |
|---|---|---|---|---|---|
| add a --verbose flag to the CLI parser | edit | **read** 87% | edit 70% | edit 95% | edit 100% |
| what did we decide about the cache earlier? | answer | answer 56% | answer 86% | answer 100% | answer 100% |
| go through every paper and summarise them | agent | agent 54% | agent 96% | agent 100% | agent 100% |
| write a script that benchmarks this endpoint | write | write 71% | write 97% | write 99% | write 100% |
| | | 9/10 | 10/10 | 10/10 | 10/10 |

Ten cases still cannot measure accuracy, and nothing here says the last two
columns are better than the second. What they do show is the confidence moving
from the fifties and seventies to near one on cases that were already right,
which is what a confidence gate is spending.

## Guidance, as measured

1. Give every tool a sentence, and spend it on the boundary with the tool next to
   it rather than on what the tool does. Worth about three points (p < 1e-4).
2. Ask for what the label means. "Which tool should be used first" and "which
   tool carries out this request" have different correct answers.
3. Do not bother with worked examples, generated disambiguation rules or option
   calibration for a closed-set routing question. All three were measured here
   and none of them moved it.
4. Check the catalogue for pairs that cannot be told apart. They cap accuracy
   whatever the prompt says, and they are visible in the specs.
5. Label options rather than fight the tokenizer. It costs nothing.
6. Gate on confidence and escalate what falls below. At 0.97 that is 78% of
   decisions at 0.989.

## Reproducing

`probes/routing/` in this repo: `build.py` and `build_hard.py` construct the two
benchmarks from a local copy of xlam-function-calling-60k, `sharpen.py` generates
the boundary rules, `run.py` and `run_hard.py` run the arm ladders, and
`names_vs_letters.py` and `probe_router.py` run the two side experiments. Start
the fork's server with `--decide-ctx 32768 --decide-seq 4` and point the scripts
at it.

A third benchmark was attempted and abandoned: predicting the tool an agent
actually used next, from a local corpus of 412 Claude Code transcripts. After
excluding the rows the harness injects as user turns — task notifications, hook
output — the corpus holds only a few hundred genuine requests and is too
concentrated on one tool to measure a multi-class router. It is recorded here
because the first count of that corpus was six thousand cases and the useful
number was two orders of magnitude smaller.
