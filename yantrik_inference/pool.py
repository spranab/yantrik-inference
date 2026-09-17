"""Pools of contexts over one set of weights.

A context is a working memory: its KV cache and scratch buffers. The weights are
read-only during inference, so N contexts can share one copy of them and run at
the same time. That is the whole reason this is cheap: adding a worker costs its
cache, not another 16 GB of model.

It also means the two kinds of work can be sized independently, which they should
be, because they are not alike:

    decide   short records, many sequences. 8k of budget split 16 ways is
             plenty, and a smaller context is measurably faster.
    chat     one long sequence. 32k so a conversation has somewhere to live.

llama.cpp divides a context's budget by its sequence count, so a decide worker's
per-question room is decide_ctx / decide_seq.
"""
from __future__ import annotations

import queue
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Generic, List, TypeVar

T = TypeVar("T")


class PoolTimeout(RuntimeError):
    pass


class Pool(Generic[T]):
    """Workers checked out one at a time. Each holds its own context, so two
    checkouts really do run concurrently."""

    def __init__(self, workers: List[T], name: str):
        self.name = name
        self._all = list(workers)
        self._free: "queue.LifoQueue[T]" = queue.LifoQueue()
        for w in workers:
            self._free.put(w)
        self.waits = 0                 # how often a caller had to queue

    def __len__(self) -> int:
        return len(self._all)

    @property
    def idle(self) -> int:
        return self._free.qsize()

    @contextmanager
    def acquire(self, timeout: float = 120.0):
        if self._free.empty():
            self.waits += 1
        try:
            w = self._free.get(timeout=timeout)
        except queue.Empty:
            raise PoolTimeout(
                f"no {self.name} worker free after {timeout:.0f}s; all "
                f"{len(self._all)} are busy. Start with a larger --{self.name}-pool."
            ) from None
        try:
            yield w
        finally:
            self._free.put(w)


@dataclass
class PoolPlan:
    """What a pool will cost before it is built, so `serve` can say so."""
    workers: int
    ctx: int
    seqs: int = 1

    @property
    def per_seq(self) -> int:
        return self.ctx // self.seqs

    @property
    def total_tokens(self) -> int:
        return self.workers * self.ctx


def build(n: int, make: Callable[[int], T], name: str, on_fail: str = "") -> Pool[T]:
    """Build n workers, stopping early if the device runs out of room rather
    than failing the whole server: a smaller pool still works."""
    from .engine import EngineError
    made: List[T] = []
    for i in range(n):
        try:
            made.append(make(i))
        except (EngineError, ValueError, MemoryError) as e:
            if not made:
                raise
            print(f"  note: only {len(made)} of {n} {name} workers fit ({e}). "
                  f"{on_fail}".rstrip())
            break
    return Pool(made, name)
