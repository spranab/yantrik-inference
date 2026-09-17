"""Model loading and the two contexts.

One set of weights, two llama.cpp contexts over them, because the two things this
serves want opposite shapes:

  decide   many sequences, each short. llama.cpp divides a context's token budget
           by its sequence count, so N typed questions need N sequences that each
           hold the record plus one question.
  chat     one sequence, long. A conversation wants the whole budget to itself.

Loading the model twice would double the weight memory (16.5 GB for a quantised
27B) and not fit on a single consumer card, so the second context is created
directly over the first model.
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Optional

# ggml type ids, for the KV cache
KV_TYPES = {"f16": None, "q8_0": 8, "q5_1": 7, "q4_0": 2}


class EngineError(RuntimeError):
    pass


@dataclass
class Limits:
    decide_ctx: int
    decide_seq: int
    chat_ctx: int
    kv_type: str
    decide_pool: int = 1
    chat_pool: int = 1

    @property
    def per_seq(self) -> int:
        return self.decide_ctx // self.decide_seq

    @property
    def cache_tokens(self) -> int:
        """Total cached tokens across every worker. Weights are shared, so the
        per-worker cost is cache plus buffers, nothing more."""
        return self.decide_pool * self.decide_ctx + self.chat_pool * self.chat_ctx

    @property
    def sequences(self) -> int:
        """Total sequences across every worker. On a hybrid (linear-attention)
        model this is the number that costs, not the context length: each
        sequence carries its own fixed-size recurrent state."""
        return self.decide_pool * self.decide_seq + self.chat_pool


def _patch_seq_max(C, n_seq: int):
    """llama-cpp-python does not expose n_seq_max, and the default is 1, which
    makes any multi-sequence batch fail with -1. Patch the default params for the
    duration of construction."""
    orig = C.llama_context_default_params

    def patched():
        p = orig()
        p.n_seq_max = n_seq
        return p

    C.llama_context_default_params = patched
    return orig


def load_model(path: str, *, n_ctx: int, n_batch: int, n_seq: int, n_gpu_layers: int = -1,
               main_gpu: int = 0, split: bool = False, kv_type: str = "q8_0",
               verbose: bool = False):
    """Load the weights and build the decide context. Returns (llm, C)."""
    if not os.path.exists(path):
        raise EngineError(f"model file not found: {path}")
    try:
        import llama_cpp
        from llama_cpp import llama_cpp as C
    except ImportError as e:  # pragma: no cover
        raise EngineError(
            "llama-cpp-python is not installed. For GPU you need a CUDA build:\n"
            "  CMAKE_ARGS='-DGGML_CUDA=on' pip install --no-binary llama-cpp-python llama-cpp-python"
        ) from e

    kv = KV_TYPES.get(kv_type)
    if kv_type not in KV_TYPES:
        raise EngineError(f"unknown kv type {kv_type!r}; choose from {sorted(KV_TYPES)}")

    orig = _patch_seq_max(C, n_seq)
    try:
        kw = dict(model_path=path, n_ctx=n_ctx, n_batch=n_batch, n_ubatch=n_batch,
                  n_gpu_layers=n_gpu_layers, logits_all=False, verbose=verbose,
                  main_gpu=main_gpu)
        if not split:
            kw["split_mode"] = 0                     # whole model on main_gpu
        if kv is not None:
            # llama.cpp refuses a quantised V cache unless flash attention is on,
            # and the wrapper sets flash_attn_type itself, so both go through it.
            kw.update(type_k=kv, type_v=kv, flash_attn=True)
        llm = llama_cpp.Llama(**kw)
    except ValueError as e:
        raise EngineError(
            f"could not create the decide context ({e}). Common causes: not enough "
            f"VRAM for {n_ctx} tokens of cache on top of the weights (try a smaller "
            f"--decide-ctx), or a model this llama.cpp build does not support."
        ) from e
    finally:
        C.llama_context_default_params = orig
    return llm, C


def second_context(llm, C, *, n_ctx: int, n_batch: int, kv_type: str = "q8_0",
                   n_seq: int = 1, verbose: bool = False):
    """Another context over the SAME weights. `n_seq` shapes it: 1 for a
    conversation, many for reading typed fields."""
    import llama_cpp
    kv = KV_TYPES.get(kv_type)
    src = llm.context_params
    p = C.llama_context_default_params()
    # The library defaults are not usable here (4 threads, 512 tokens) and the
    # struct holds pointers, so copy.copy cannot clone it: copy field by field.
    for name, _ in src._fields_:
        try:
            setattr(p, name, getattr(src, name))
        except Exception:                            # noqa: BLE001
            pass
    p.n_ctx, p.n_batch, p.n_ubatch, p.n_seq_max = n_ctx, n_batch, min(n_batch, 512), n_seq
    if kv is not None:
        p.type_k = p.type_v = kv
        p.flash_attn_type = C.LLAMA_FLASH_ATTN_TYPE_ENABLED
    try:
        return llama_cpp._internals.LlamaContext(model=llm._model, params=p, verbose=verbose)
    except ValueError as e:
        raise EngineError(
            f"could not create a {n_ctx}-token context ({e}). The weights are shared "
            f"but each worker needs its own cache and compute buffers; use a smaller "
            f"context, a smaller pool, or a smaller --n-batch."
        ) from e


def context_cost(llm, n_ctx: int, n_seq: int, kv_type: str = "q8_0") -> dict:
    """Estimate what one context will cost, in MiB, from the model's metadata.

    Worth doing because the intuition from dense models is wrong here. On a
    hybrid model most layers are linear attention, and each of those keeps a
    fixed-size recurrent state PER SEQUENCE that does not depend on the context
    length at all. Measured on Qwen3.8-27B: 150 MiB per sequence against 136 MiB
    per 4k of context, so 32 sequences at 8k cost more than one sequence at 128k.
    """
    md = llm.metadata
    def num(*keys, default=0):
        for k in keys:
            for full in (k, *(f"{p}.{k}" for p in ("qwen35", "qwen3", "llama", "general"))):
                if full in md:
                    try:
                        return int(md[full])
                    except (TypeError, ValueError):
                        pass
        return default

    bytes_per = {"f16": 2.0, "q8_0": 1.0625, "q5_1": 0.75, "q4_0": 0.5625}.get(kv_type, 2.0)
    layers = num("block_count", default=0)
    interval = num("full_attention_interval", default=1) or 1
    full_layers = max(1, layers // interval) if layers else 0
    kv_heads = num("attention.head_count_kv", default=0)
    k_len = num("attention.key_length", default=0)
    v_len = num("attention.value_length", default=0)
    kv_mib = full_layers * n_ctx * kv_heads * (k_len + v_len) * bytes_per / (1 << 20)

    ssm_layers = max(0, layers - full_layers)
    state = num("ssm.state_size")
    inner, conv = num("ssm.inner_size"), num("ssm.conv_kernel")
    # the recurrent state is inner_size x state_size in f32, plus the conv window;
    # it is per sequence and does not grow with the context
    per_seq = (inner * state * 4 + inner * conv * 4) / (1 << 20) if inner and state else 0.0
    rs_mib = ssm_layers * n_seq * per_seq
    return dict(kv_mib=round(kv_mib), recurrent_mib=round(rs_mib),
                per_sequence_mib=round(per_seq * ssm_layers, 1),
                hybrid=ssm_layers > 0)


def chat_parts(llm):
    """The model's chat template split around a sentinel, so a record can be
    prefilled once and each question appended to it.

    A plain completion prompt is not good enough: on Qwen3.5-0.8B it made the
    model answer "yes" to 100% of boolean fields (0.35 accuracy, below the 0.60
    majority baseline). With the template it scored 0.89.
    """
    sent = "<<<SPLIT>>>"
    tmpl = llm.metadata.get("tokenizer.chat_template")
    if tmpl:
        try:
            from jinja2 import Environment
            env = Environment()
            env.globals["strftime_now"] = lambda fmt: ""
            rendered = env.from_string(tmpl).render(
                messages=[{"role": "user", "content": sent}], add_generation_prompt=True,
                enable_thinking=False, tools=None)
            if sent in rendered:
                head, tail = rendered.split(sent)
                return head, tail
        except Exception:                            # noqa: BLE001
            pass
    # ChatML with the thinking block closed, which is what Qwen3.x emits for
    # enable_thinking=False
    return "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Memory:
    """The cache-management calls, which llama.cpp renamed in 0.3.3x.

    The old `llama_kv_self_*` names still exist as no-ops in some builds, so
    calling them silently leaks sequences until the cache has no free slot and
    llama_decode returns 1. Resolve the family once, here.
    """

    def __init__(self, C, ctx):
        self.C, self.ctx = C, ctx
        self._handle = getattr(C, "llama_get_memory", lambda _c: None)(ctx)
        self._new = hasattr(C, "llama_memory_clear")

    def clear(self):
        C = self.C
        if self._new:
            return C.llama_memory_clear(self._handle, True)
        return C.llama_kv_self_clear(self.ctx)

    def seq_cp(self, src: int, dst: int, p0: int = 0, p1: int = -1):
        """Share a prefix with another sequence. This is cell tagging, not a data
        copy, so the prefix is stored once however many sequences read it. Some
        builds assert on partial ranges, hence the whole-sequence default."""
        C = self.C
        if self._new:
            return C.llama_memory_seq_cp(self._handle, src, dst, p0, p1)
        return C.llama_kv_self_seq_cp(self.ctx, src, dst, p0, p1)

    def seq_rm(self, seq: int, p0: int = -1, p1: int = -1):
        C = self.C
        if self._new:
            return C.llama_memory_seq_rm(self._handle, seq, p0, p1)
        return C.llama_kv_self_seq_rm(self.ctx, seq, p0, p1)


def decode(C, ctx, tokens, positions, seq_ids, want_logits, n_vocab):
    """One llama_decode over an explicitly built batch. Returns {index: logits}."""
    import numpy as np
    n = len(tokens)
    batch = C.llama_batch_init(n, 0, 1)
    try:
        for i in range(n):
            batch.token[i] = int(tokens[i])
            batch.pos[i] = int(positions[i])
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = int(seq_ids[i])
            batch.logits[i] = 1 if want_logits[i] else 0
        batch.n_tokens = n
        rc = C.llama_decode(ctx, batch)
        if rc == 1:
            raise EngineError(
                "the cache has no free slot for this batch. The record plus one "
                "question must fit in the per-sequence budget (context / sequences)."
            )
        if rc != 0:
            raise EngineError(f"llama_decode returned {rc}")
        out = {}
        for i in range(n):
            if want_logits[i]:
                p = C.llama_get_logits_ith(ctx, i)
                out[i] = np.ctypeslib.as_array(
                    ctypes.cast(p, ctypes.POINTER(ctypes.c_float)), (n_vocab,)).copy()
        return out
    finally:
        C.llama_batch_free(batch)
