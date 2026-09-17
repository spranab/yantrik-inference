"""The HTTP server: typed decisions and an OpenAI-compatible chat, one model."""
from __future__ import annotations

import json
import random
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from .chat import ChatEngine
from . import pool
from .engine import Limits, chat_parts, context_cost, load_model, second_context
from .reader import Field, FieldReader
from .tasks import make_case
from .ui import PAGE


class Service:
    """Everything the handlers need.

    One copy of the weights, two pools of contexts over them. Workers in a pool
    run concurrently because each has its own cache; the pools are sized
    separately because deciding and generating want different shapes.
    """

    def __init__(self, model_path: str, limits: Limits, *, n_batch: int = 2048,
                 n_gpu_layers: int = -1, main_gpu: int = 0, split: bool = False,
                 sample_fields: int = 10, verbose: bool = False):
        t0 = time.time()
        # Batch size only moves the compute buffer (about 0.5 GB at 512 against
        # 2.0 GB at 2048 on a 27B), so size each pool to the largest batch it will
        # actually submit. The dominant cost is elsewhere: see context_cost.
        decide_batch = min(n_batch, max(limits.per_seq, limits.decide_seq * 64, 512))
        chat_batch = min(n_batch, 512)
        self.decide_batch, self.chat_batch = decide_batch, chat_batch
        self.llm, self.C = load_model(
            model_path, n_ctx=limits.decide_ctx, n_batch=decide_batch,
            n_seq=limits.decide_seq, n_gpu_layers=n_gpu_layers, main_gpu=main_gpu,
            split=split, kv_type=limits.kv_type, verbose=verbose)
        head, tail = chat_parts(self.llm)
        self.head, self.tail = head, tail
        self.cost = dict(
            decide=context_cost(self.llm, limits.decide_ctx, limits.decide_seq, limits.kv_type),
            chat=context_cost(self.llm, limits.chat_ctx, 1, limits.kv_type))

        def make_reader(i: int) -> FieldReader:
            # worker 0 reuses the context the model was loaded with; the rest get
            # their own, same shape
            ctx = None
            if i:
                ctx = second_context(self.llm, self.C, n_ctx=limits.decide_ctx,
                                     n_batch=decide_batch, n_seq=limits.decide_seq,
                                     kv_type=limits.kv_type, verbose=verbose).ctx
            return FieldReader(self.llm, self.C, head, tail, limits.decide_seq,
                               limits.per_seq, ctx=ctx)

        def make_chat(i: int) -> ChatEngine:
            return ChatEngine(self.llm, self.C, limits.chat_ctx, chat_batch,
                              kv_type=limits.kv_type, verbose=verbose)

        self.chats = pool.build(limits.chat_pool, make_chat, "chat",
                                "Chat requests will queue beyond that.")
        self.readers = pool.build(limits.decide_pool, make_reader, "decide",
                                  "Decide requests will queue beyond that.")
        self.limits = limits
        self.load_s = round(time.time() - t0, 1)
        self.name = model_path.replace("\\", "/").rsplit("/", 1)[-1]
        self.sample_fields = sample_fields
        self.where = ("split across GPUs" if split else f"GPU {main_gpu}") \
            if n_gpu_layers != 0 else "CPU"

    @property
    def reader(self) -> FieldReader:
        """Any worker, for prompt measurements that touch no cache."""
        return self.readers._all[0]


SERVICE: Optional[Service] = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "yantrik-inference"

    def log_message(self, *a):                        # quiet by default
        pass

    # -- plumbing ------------------------------------------------------------
    def _send(self, code: int, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(b)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    # -- GET -----------------------------------------------------------------
    def do_GET(self):
        s = SERVICE
        if self.path in ("/", "/index.html"):
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if self.path == "/health":
            return self._send(200, json.dumps(dict(
                ok=True, model=s.name,
                decide=dict(workers=len(s.readers), idle=s.readers.idle, queued=s.readers.waits),
                chat=dict(workers=len(s.chats), idle=s.chats.idle, queued=s.chats.waits))))
        if self.path in ("/v1/models", "/models"):
            return self._send(200, json.dumps(dict(object="list", data=[
                dict(id=s.name, object="model", owned_by="local",
                     created=int(time.time()))])))
        if self.path == "/api/info":
            text, qs = make_case(random.Random(7))
            qs = qs[: s.sample_fields]
            return self._send(200, json.dumps(dict(
                model=s.name, where=s.where, load_s=s.load_s,
                per_seq=s.limits.per_seq, max_fields=s.limits.decide_seq,
                chat_ctx=s.limits.chat_ctx, kv=s.limits.kv_type, sample_record=text,
                decide_pool=len(s.readers), chat_pool=len(s.chats),
                sample_questions=[dict(q=q, opts=list(o)) for q, o, _ in qs])))
        return self._send(404, json.dumps(dict(error="not found")))

    # -- POST ----------------------------------------------------------------
    def do_POST(self):
        try:
            req = self._body()
        except Exception as e:                        # noqa: BLE001
            return self._send(400, json.dumps(dict(error=f"bad JSON: {e}")))
        if self.path in ("/v1/decide", "/api/decide"):
            return self._decide(req, as_json=False)
        if self.path == "/api/decide_json":
            return self._decide(req, as_json=True)
        if self.path in ("/v1/chat/completions", "/chat/completions"):
            return self._chat(req)
        return self._send(404, json.dumps(dict(error="not found")))

    def _decide(self, req, as_json: bool):
        s = SERVICE
        record = (req.get("record") or req.get("text") or "").strip()
        raw = req.get("questions") or []
        fields = []
        for q in raw:
            if isinstance(q, str):
                f = Field.parse(q)
                if f:
                    fields.append(f)
            elif q.get("q") and len(q.get("opts") or q.get("options") or []) > 1:
                fields.append(Field(q["q"], tuple(q.get("opts") or q["options"])))
        if not record or not fields:
            return self._send(400, json.dumps(dict(
                error="need `record` and `questions`; a question is either "
                      "'text | opt/opt' or {\"q\":…, \"opts\":[…]}")))
        try:
            with s.readers.acquire() as reader:
                t0 = time.time()
                if not as_json:
                    answers = reader.read(record, fields)
                    dt = time.time() - t0
                    return self._send(200, json.dumps(dict(
                        seconds=round(dt, 4), model=s.name,
                        answers=[dict(question=a.question, answer=a.answer,
                                      confidence=round(a.confidence, 4)) for a in answers])))
                prompt_len = len(reader.tok(reader.head + record, bos=True)) + 40 * len(fields)
                room = s.limits.per_seq - prompt_len - 16
                if room < 120:
                    return self._send(200, json.dumps(dict(
                        seconds=0.0, answers=[None] * len(fields),
                        error=f"no room to write the object: the prompt is about "
                              f"{prompt_len} tokens and each sequence holds "
                              f"{s.limits.per_seq}. Use fewer questions.")))
                vals, _ = reader.read_as_json(record, fields, max_new=min(room, 900))
                dt = time.time() - t0
                return self._send(200, json.dumps(dict(
                    seconds=round(dt, 4), answers=vals or [None] * len(fields),
                    error=None if vals else "the model's JSON did not parse")))
        except ValueError as e:
            return self._send(400, json.dumps(dict(error=str(e))))
        except Exception as e:                        # noqa: BLE001
            return self._send(500, json.dumps(dict(error=f"{type(e).__name__}: {e}")))

    def _chat(self, req):
        s = SERVICE
        msgs = req.get("messages") or []
        if not msgs and req.get("prompt"):
            msgs = [dict(role="user", content=req["prompt"])]
        if not msgs:
            return self._send(400, json.dumps(dict(error="`messages` is required")))
        kw = dict(max_tokens=int(req.get("max_tokens") or 512),
                  temperature=float(req.get("temperature", 0.7)),
                  top_p=float(req.get("top_p", 0.95)))
        cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
        base = dict(id=cid, created=created, model=s.name)
        try:
            if not req.get("stream"):
                with s.chats.acquire() as engine:
                    text = "".join(engine.stream(msgs, **kw))
                return self._send(200, json.dumps(dict(
                    **base, object="chat.completion",
                    choices=[dict(index=0, finish_reason="stop",
                                  message=dict(role="assistant", content=text))])))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            def chunk(delta, finish=None):
                payload = dict(**base, object="chat.completion.chunk",
                               choices=[dict(index=0, delta=delta, finish_reason=finish)])
                self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                self.wfile.flush()

            chunk(dict(role="assistant"))
            with s.chats.acquire() as engine:
                for piece in engine.stream(msgs, **kw):
                    chunk(dict(content=piece))
            chunk({}, "stop")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:                        # noqa: BLE001
            try:
                self._send(500, json.dumps(dict(error=f"{type(e).__name__}: {e}")))
            except Exception:                         # noqa: BLE001
                pass


def serve(service: Service, host: str, port: int):
    # stdout is block-buffered when it is not a terminal, so a server that
    # prints its configuration and then blocks would show nothing at all
    # until it exits. Flush every line.
    import functools
    print = functools.partial(__builtins__['print'] if isinstance(__builtins__, dict)
                              else __builtins__.print, flush=True)
    global SERVICE
    SERVICE = service
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"{service.name} ready in {service.load_s}s on {service.where}")
    d, c = service.cost["decide"], service.cost["chat"]
    print(f"  decide  {len(service.readers)} worker(s) x {service.limits.decide_seq} "
          f"sequences x {service.limits.per_seq} tokens "
          f"(~{d['kv_mib'] + d['recurrent_mib']} MiB each)")
    print(f"  chat    {len(service.chats)} worker(s) x {service.limits.chat_ctx} tokens "
          f"(~{c['kv_mib'] + c['recurrent_mib']} MiB each)")
    if d["hybrid"]:
        # the counter-intuitive part, said out loud because sizing depends on it
        print(f"  note    this model keeps a per-sequence recurrent state: each "
              f"sequence costs ~{d['per_sequence_mib']:.0f} MiB whatever the context "
              f"length, so --decide-seq is the expensive knob, not --decide-ctx")
    print(f"  cache   {service.limits.kv_type}")
    print(f"  open    http://{'localhost' if host in ('0.0.0.0', '') else host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
        httpd.shutdown()
