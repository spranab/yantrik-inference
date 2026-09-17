"""Command line: serve, decide, bench, pull."""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__
from .engine import EngineError, Limits


def _model_args(p):
    p.add_argument("--model", "-m", required=True,
                   help="path to a .gguf file, or a Hugging Face repo id "
                        "(optionally repo:filename-substring) to pull if missing")
    p.add_argument("--decide-ctx", type=int, default=8192,
                   help="token budget per decide worker (default 8192; records are "
                        "short and a smaller context is measurably faster)")
    p.add_argument("--decide-seq", type=int, default=16,
                   help="how many fields can be read at once; the budget is "
                        "divided by this (default 24)")
    p.add_argument("--chat-ctx", type=int, default=32768,
                   help="token budget for a conversation (default 32768)")
    p.add_argument("--decide-pool", type=int, default=2,
                   help="how many decide requests can run at once (default 2)")
    p.add_argument("--chat-pool", type=int, default=1,
                   help="how many chat requests can run at once (default 1)")
    p.add_argument("--kv-type", default="q8_0", choices=("f16", "q8_0", "q5_1", "q4_0"),
                   help="KV cache precision (default q8_0, which halves its memory "
                        "with no measured accuracy cost)")
    p.add_argument("--n-batch", type=int, default=2048)
    p.add_argument("--gpu-layers", type=int, default=-1,
                   help="-1 for all on GPU, 0 for CPU only")
    p.add_argument("--main-gpu", type=int, default=0)
    p.add_argument("--split", action="store_true", help="split layers across GPUs")
    p.add_argument("--no-guard", action="store_true",
                   help="do not wrap the record as untrusted data; without the guard, "
                        "text inside a record can steer the answer")
    p.add_argument("--verbose", action="store_true")


def resolve_model(spec: str) -> str:
    """A local path, or `repo` / `repo:substring` pulled from Hugging Face."""
    if os.path.exists(spec):
        return spec
    if "/" not in spec:
        raise EngineError(f"no such file: {spec}")
    repo, _, want = spec.partition(":")
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as e:                          # pragma: no cover
        raise EngineError("huggingface_hub is needed to pull models: "
                          "pip install huggingface_hub") from e
    files = [f for f in list_repo_files(repo) if f.lower().endswith(".gguf")]
    if not files:
        raise EngineError(f"{repo} has no .gguf files")
    if want:
        files = [f for f in files if want.lower() in f.lower()] or files
    else:
        pref = [f for f in files if "q4_k_m" in f.lower()]
        files = pref or files
    files.sort(key=len)
    if any(f.lower().endswith(("-00002-of-00002.gguf", "-00002-of-00003.gguf")) for f in files):
        files = [f for f in files if "00001-of-" in f] or files
    print(f"pulling {repo} :: {files[0]}")
    return hf_hub_download(repo, files[0])


def cmd_serve(args) -> int:
    from .server import Service, serve
    limits = Limits(args.decide_ctx, args.decide_seq, args.chat_ctx, args.kv_type,
                    decide_pool=args.decide_pool, chat_pool=args.chat_pool)
    if limits.per_seq < 256:
        print(f"warning: each sequence gets only {limits.per_seq} tokens "
              f"({args.decide_ctx} / {args.decide_seq}), which may be too few for a "
              f"record plus a question", file=sys.stderr)
    svc = Service(resolve_model(args.model), limits, n_batch=args.n_batch,
                  n_gpu_layers=args.gpu_layers, main_gpu=args.main_gpu,
                  split=args.split, verbose=args.verbose, guard=not args.no_guard)
    serve(svc, args.host, args.port)
    return 0


def _text_or_file(value: str, what: str) -> str:
    """Accept either literal text or a path. A value that looks like a filename
    but does not exist is an error, not a record: silently treating a typo'd
    path as the text produces confident answers about the filename."""
    if os.path.exists(value):
        return open(value, encoding="utf-8").read()
    looks_like_path = (" " not in value.strip()
                       and ("/" in value or "\\" in value or "." in value.rsplit(" ", 1)[-1]))
    if looks_like_path:
        raise EngineError(f"--{what}: no such file {value!r} (and it looks like a path, "
                          f"not text). Pass the text directly, or fix the path.")
    return value


def cmd_decide(args) -> int:
    """One-shot typed decisions from the command line, no server."""
    from .engine import chat_parts, load_model
    from .reader import Field, FieldReader
    record = _text_or_file(args.record, "record")
    raw_ask = _text_or_file(args.ask, "ask")
    src = raw_ask.splitlines() if "\n" in raw_ask else raw_ask.split(";")
    fields = [f for f in (Field.parse(line) for line in src) if f]
    if not fields:
        print("no questions parsed; use 'question | opt/opt' per line", file=sys.stderr)
        return 2
    limits = Limits(args.decide_ctx, max(args.decide_seq, len(fields)),
                    args.chat_ctx, args.kv_type)
    llm, C = load_model(resolve_model(args.model), n_ctx=limits.decide_ctx,
                        n_batch=args.n_batch, n_seq=limits.decide_seq,
                        n_gpu_layers=args.gpu_layers, main_gpu=args.main_gpu,
                        split=args.split, kv_type=args.kv_type, verbose=args.verbose)
    head, tail = chat_parts(llm)
    reader = FieldReader(llm, C, head, tail, limits.decide_seq, limits.per_seq,
                         guard=not args.no_guard)
    answers = reader.read(record, fields)
    if args.json:
        print(json.dumps([dict(question=a.question, answer=a.answer,
                               confidence=round(a.confidence, 4)) for a in answers], indent=1))
    else:
        w = max(len(a.answer) for a in answers)
        for a in answers:
            print(f"  {a.answer:<{w}}  {a.confidence * 100:5.1f}%  {a.question}")
    return 0


def cmd_bench(args) -> int:
    from .bench import run
    limits = Limits(args.decide_ctx, max(args.decide_seq, args.fields),
                    args.chat_ctx, args.kv_type)
    run(resolve_model(args.model), cases=args.cases, fields=args.fields, limits=limits,
        n_batch=args.n_batch, n_gpu_layers=args.gpu_layers, main_gpu=args.main_gpu,
        split=args.split, seed=args.seed, with_json=not args.no_json, out=args.out)
    return 0


def cmd_pull(args) -> int:
    print(resolve_model(args.model))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="yantrik-inference",
        description="Typed decisions and chat from one loaded model. Decisions are "
                    "read from a single forward pass instead of being generated.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the HTTP server and the web page")
    _model_args(s)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8020)
    s.set_defaults(fn=cmd_serve)

    d = sub.add_parser("decide", help="answer typed questions about a record, once")
    _model_args(d)
    d.add_argument("--record", "-r", required=True, help="a file, or the text itself")
    d.add_argument("--ask", "-a", required=True,
                   help="a file of 'question | opt/opt' lines, or the same separated by ';'")
    d.add_argument("--json", action="store_true", help="print JSON instead of a table")
    d.set_defaults(fn=cmd_decide)

    b = sub.add_parser("bench", help="reproduce the timings on your own hardware")
    _model_args(b)
    b.add_argument("--cases", type=int, default=10)
    b.add_argument("--fields", type=int, default=28)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--no-json", action="store_true", help="skip the JSON-generation row")
    b.add_argument("--out", default=None, help="write the results as JSON")
    b.set_defaults(fn=cmd_bench)

    u = sub.add_parser("pull", help="download a GGUF from Hugging Face and print its path")
    u.add_argument("--model", "-m", required=True)
    u.set_defaults(fn=cmd_pull)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except EngineError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
