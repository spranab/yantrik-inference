"""Generation over the second context.

This is ordinary autoregressive decoding and runs at ordinary speed: on a
quantised 27B on one RTX 3090 Ti, about 24 tokens/s with a one-second first
token. It is here so that one loaded copy of the weights serves both a chat and
the decide endpoint, not because the typed-field trick makes generation faster.
It does not, and cannot: a reply of N tokens needs N sequential passes.
"""
from __future__ import annotations

from typing import Iterator, List, Sequence

from .engine import Memory, decode, second_context


class ChatEngine:
    def __init__(self, llm, C, n_ctx: int, n_batch: int, kv_type: str = "q8_0",
                 verbose: bool = False, lctx=None):
        """`lctx` lets a pool supply an already-built context; otherwise one is
        created over the same weights."""
        self.llm, self.C = llm, C
        self.lctx = lctx if lctx is not None else second_context(
            llm, C, n_ctx=n_ctx, n_batch=n_batch, kv_type=kv_type, verbose=verbose)
        self.ctx = self.lctx.ctx
        self.mem = Memory(C, self.ctx)
        self.n_ctx, self.n_vocab = n_ctx, llm.n_vocab()
        self.tmpl = llm.metadata.get("tokenizer.chat_template")
        self.eos = {llm.token_eos()}
        for t in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
            ids = llm.tokenize(t.encode(), add_bos=False, special=True)
            if len(ids) == 1:
                self.eos.add(ids[0])

    def render(self, messages: Sequence[dict]) -> str:
        if self.tmpl:
            try:
                from jinja2 import Environment
                env = Environment()
                env.globals["strftime_now"] = lambda fmt: ""
                return env.from_string(self.tmpl).render(
                    messages=list(messages), add_generation_prompt=True,
                    enable_thinking=False, tools=None)
            except Exception:                        # noqa: BLE001
                pass
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return out + "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    @staticmethod
    def _sample(logits, temperature: float, top_p: float, rng):
        import numpy as np
        if temperature <= 0:
            return int(logits.argmax())
        z = logits.astype(np.float64) / max(temperature, 1e-5)
        z -= z.max()
        p = np.exp(z); p /= p.sum()
        idx = np.argsort(-p)
        keep = int(np.searchsorted(np.cumsum(p[idx]), top_p)) + 1
        idx = idx[:max(keep, 1)]
        q = p[idx] / p[idx].sum()
        return int(rng.choice(idx, p=q))

    def stream(self, messages: Sequence[dict], max_tokens: int = 512,
               temperature: float = 0.7, top_p: float = 0.95) -> Iterator[str]:
        """Yield text pieces. A multi-byte character can straddle two tokens, so
        bytes are accumulated and only fully decoded text is emitted."""
        import numpy as np
        rng = np.random.default_rng()
        ids: List[int] = self.llm.tokenize(self.render(messages).encode(),
                                           add_bos=True, special=True)
        budget = self.n_ctx - max_tokens - 8
        if budget < 64:
            raise ValueError(f"max_tokens={max_tokens} leaves no room in a "
                             f"{self.n_ctx}-token context")
        if len(ids) > budget:
            ids = ids[:1] + ids[-(budget - 1):]      # keep BOS, drop the middle
        self.mem.clear()
        got = decode(self.C, self.ctx, ids, range(len(ids)), [0] * len(ids),
                     [i == len(ids) - 1 for i in range(len(ids))], self.n_vocab)
        logits = got[len(ids) - 1]
        pos, buf, emitted = len(ids), b"", ""
        for _ in range(max_tokens):
            tok = self._sample(logits, temperature, top_p, rng)
            if tok in self.eos:
                break
            buf += self.llm.detokenize([tok])
            text = buf.decode("utf-8", errors="ignore")
            if len(text) > len(emitted):
                yield text[len(emitted):]
                emitted = text
            got = decode(self.C, self.ctx, [tok], [pos], [0], [True], self.n_vocab)
            logits = got[0]
            pos += 1
