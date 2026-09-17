"""yantrik-inference — typed decisions and chat from one loaded model.

A decision with a known set of answers does not need the model to write anything.
Prefill the record once, share that prefix across one sequence per question, and
read every answer from a single batched forward pass, restricted to each field's
allowed tokens. The output is valid by construction because it is never parsed.

    from yantrik_inference import Field, open_model

    reader, chat = open_model("model.gguf")
    for a in reader.read(record, [Field.parse("Is the card present? | yes/no")]):
        print(a.answer, a.confidence)
"""
from __future__ import annotations

__version__ = "0.1.0"

from .engine import EngineError, Limits, chat_parts, load_model  # noqa: F401
from .reader import Answer, Field, FieldReader  # noqa: F401


def open_model(path: str, *, decide_ctx: int = 24576, decide_seq: int = 24,
               chat_ctx: int = 16384, kv_type: str = "q8_0", n_batch: int = 2048,
               n_gpu_layers: int = -1, main_gpu: int = 0, split: bool = False,
               with_chat: bool = True, guard: bool = True):
    """Load once, return (FieldReader, ChatEngine | None)."""
    from .chat import ChatEngine
    llm, C = load_model(path, n_ctx=decide_ctx, n_batch=n_batch, n_seq=decide_seq,
                        n_gpu_layers=n_gpu_layers, main_gpu=main_gpu, split=split,
                        kv_type=kv_type)
    head, tail = chat_parts(llm)
    reader = FieldReader(llm, C, head, tail, decide_seq, decide_ctx // decide_seq,
                         guard=guard)
    chat = ChatEngine(llm, C, chat_ctx, n_batch, kv_type=kv_type) if with_chat else None
    return reader, chat


__all__ = ["Answer", "EngineError", "Field", "FieldReader", "Limits",
           "chat_parts", "load_model", "open_model", "__version__"]
