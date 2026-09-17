"""Typed field reading: many decisions from one forward pass.

Given a record and N typed questions, prefill the record once, share that prefix
with N sequences, append each question, and read every answer from a single
batched pass. Each answer is the argmax over that field's allowed tokens only, so
the output is valid by construction and never parsed.

What this is worth, measured on Qwen3.8-27B (Q4_K_M, one RTX 3090 Ti, 28 fields):

    generate the answers as JSON   10.17 s
    ask one field at a time        2.24 s
    this                           2.08 s

The large win is not generating; the batched pass is the clean way to get the
prefix sharing that the middle row also has. See `yantrik-inference bench`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .engine import Memory, decode


@dataclass
class Field:
    """One typed question. `options` must have at least two entries; the answer is
    whichever option's first token the model ranks highest."""
    question: str
    options: Tuple[str, ...]

    @staticmethod
    def parse(line: str) -> "Field | None":
        """`question | opt/opt/opt`, or just a question for yes/no."""
        line = line.strip()
        if not line:
            return None
        if "|" not in line:
            return Field(line, ("yes", "no"))
        q, _, opts = line.rpartition("|")
        chosen = tuple(o.strip() for o in opts.split("/") if o.strip())
        if len(chosen) < 2:
            return None
        return Field(q.strip(), chosen)


@dataclass
class Answer:
    question: str
    answer: str
    confidence: float


class FieldReader:
    def __init__(self, llm, C, head: str, tail: str, max_fields: int, per_seq: int,
                 ctx=None):
        """`ctx` selects which context to read through. It defaults to the one
        the model was loaded with; a pool passes its own, so several readers can
        share the weights and work at the same time."""
        self.llm, self.C = llm, C
        self.head, self.tail = head, tail
        self.max_fields, self.per_seq = max_fields, per_seq
        self.ctx = ctx if ctx is not None else llm._ctx.ctx
        self.mem = Memory(C, self.ctx)
        self.n_vocab = llm.n_vocab()
        self._first: dict[str, int] = {}

    # -- prompt pieces -------------------------------------------------------
    def tok(self, s: str, bos: bool = False) -> List[int]:
        return self.llm.tokenize(s.encode(), add_bos=bos, special=True)

    def first_tokens(self, option: str) -> List[int]:
        """Every plausible first token for this option.

        Which one the model actually uses depends on the chat template. ChatML
        ends the prompt with a newline, so the model writes `no`; a template that
        ends mid-line expects ` no`. Scoring only one variant reads the wrong
        token and the ranking between options becomes arbitrary: on
        Llama-3.2-3B, scoring only ` yes`/` no` gave 99%-confident wrong answers
        while the model's actual top token was `no` at a much higher logit.

        So score every variant and take each option's best.
        """
        if option not in self._first:
            seen, out = set(), []
            for cand in (option, " " + option, option.capitalize(),
                         " " + option.capitalize(), option.upper()):
                ids = self.tok(cand)
                if ids and ids[0] not in seen:
                    seen.add(ids[0]); out.append(ids[0])
            self._first[option] = out
        return self._first[option]

    def first_token(self, option: str) -> int:
        """The most likely single id, kept for callers that want one."""
        return self.first_tokens(option)[0]

    def check_options(self, fields: Sequence[Field]) -> List[str]:
        """Options within a field must be distinguishable by their first token.
        'approve' and 'approved' are not, and would silently alias."""
        problems = []
        for f in fields:
            first = {}
            for o in f.options:
                for t in self.first_tokens(o):
                    if t in first and first[t] != o:
                        problems.append(
                            f"{f.question!r}: {first[t]!r} and {o!r} start with the "
                            f"same token, so they cannot be told apart")
                        break
                    first.setdefault(t, o)
        return problems

    def suffix(self, f: Field) -> str:
        return (f"\n\nQuestion: {f.question}\nAnswer with exactly one word from: "
                f"{', '.join(f.options)}" + self.tail)

    # -- the read ------------------------------------------------------------
    def read(self, record: str, fields: Sequence[Field]) -> List[Answer]:
        if not fields:
            return []
        if len(fields) > self.max_fields:
            raise ValueError(
                f"{len(fields)} questions but this context holds {self.max_fields} "
                f"sequences; start the server with a larger --decide-seq")
        problems = self.check_options(fields)
        if problems:
            raise ValueError("; ".join(problems))
        self.mem.clear()
        prefix = self.tok(self.head + record, bos=True)
        P = len(prefix)
        suffixes = [self.tok(self.suffix(f)) for f in fields]
        longest = P + max(len(s) for s in suffixes)
        if longest > self.per_seq:
            raise ValueError(
                f"the record plus a question is {longest} tokens but each sequence "
                f"holds {self.per_seq}. Use a shorter record, fewer questions "
                f"(--decide-seq), or a larger --decide-ctx.")
        decode(self.C, self.ctx, prefix, range(P), [0] * P, [False] * P, self.n_vocab)
        for i in range(1, len(suffixes)):
            self.mem.seq_cp(0, i)
        toks, pos, seqs, want = [], [], [], []
        for i, s in enumerate(suffixes):
            for j, t in enumerate(s):
                toks.append(t); pos.append(P + j); seqs.append(i)
                want.append(j == len(s) - 1)
        got = decode(self.C, self.ctx, toks, pos, seqs, want, self.n_vocab)
        return self._pick([got[i] for i in sorted(got)], fields)

    def read_sequential(self, record: str, fields: Sequence[Field],
                        reuse_prefix: bool = True) -> List[Answer]:
        """The same answers, one field per pass. Kept for `bench`."""
        self.mem.clear()
        prefix = self.tok(self.head + record, bos=True)
        P = len(prefix)
        if reuse_prefix:
            decode(self.C, self.ctx, prefix, range(P), [0] * P, [False] * P, self.n_vocab)
        rows = []
        for f in fields:
            s = self.tok(self.suffix(f))
            if reuse_prefix:
                self.mem.seq_cp(0, 1)
                got = decode(self.C, self.ctx, s, [P + j for j in range(len(s))],
                             [1] * len(s), [j == len(s) - 1 for j in range(len(s))],
                             self.n_vocab)
                rows.append(got[len(s) - 1])
                self.mem.seq_rm(1)
            else:
                self.mem.clear()
                full = prefix + s
                got = decode(self.C, self.ctx, full, range(len(full)), [0] * len(full),
                             [j == len(full) - 1 for j in range(len(full))], self.n_vocab)
                rows.append(got[len(full) - 1])
        return self._pick(rows, fields)

    def _pick(self, rows, fields: Sequence[Field]) -> List[Answer]:
        import numpy as np
        out = []
        for logits, f in zip(rows, fields):
            # each option scores as its best spelling at this position
            v = np.array([max(float(logits[i]) for i in self.first_tokens(o))
                          for o in f.options], dtype=np.float64)
            e = np.exp(v - v.max()); p = e / e.sum()
            j = int(p.argmax())
            out.append(Answer(f.question, f.options[j], float(p[j])))
        return out

    # -- the baseline everyone actually runs ---------------------------------
    def json_prompt(self, record: str, fields: Sequence[Field]) -> str:
        keys = [f"f{i}" for i in range(len(fields))]
        qlist = "\n".join(f'  "{k}": {f.question} ({"/".join(f.options)})'
                          for k, f in zip(keys, fields))
        schema = ", ".join(f'"{k}": "<{"/".join(f.options)}>"' for k, f in zip(keys, fields))
        return (f"{record}\n\nAnswer every question about the record above as one JSON "
                f"object, one key per question, values lowercase, no other text.\n"
                f"Questions:\n{qlist}\nFormat: {{{schema}}}")

    def read_as_json(self, record: str, fields: Sequence[Field], max_new: int = 700,
                     generate=None):
        """Ask the model to write the whole object and parse it. Returns
        (answers or None, raw text). Reported with its validity rate in `bench`,
        never assumed to parse.

        `generate` lets the caller supply a generator bound to a different
        context. The decide context divides its token budget by its sequence
        count, which is often too little to write an object for many fields; a
        one-sequence context has room.
        """
        import json
        keys = [f"f{i}" for i in range(len(fields))]
        body = self.json_prompt(record, fields)
        if generate is not None:
            txt = generate(body, max_new)
        else:
            self.mem.clear()
            out = self.llm.create_completion(self.head + body + self.tail,
                                             max_tokens=max_new, temperature=0.0)
            txt = out["choices"][0]["text"]
        i, j = txt.find("{"), txt.rfind("}")
        try:
            obj = json.loads(txt[i:j + 1])
        except Exception:                            # noqa: BLE001
            return None, txt
        vals = []
        for k, f in zip(keys, fields):
            v = str(obj.get(k, "")).strip().lower()
            vals.append(v if v in f.options else None)
        return vals, txt
