"""Have the model write the tool boundaries once, from the specs alone.

A one-line description is written without knowing which other tools exist. When
two of them overlap, the thing a router needs is the line between them, and
nobody wrote it. This asks the same model for it once per catalogue, using only
the specs -- no labelled cases -- and the result is cached in the preamble, so
it costs nothing per request.
"""
import json, io, urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
CL  = json.load(io.open("bench_hard.json", encoding="utf-8"))

ASK = """These tools are easy to confuse with each other:

{specs}

For each tool write one short line saying when to choose it *instead of* the others, keyed on what the request would contain. Be concrete about the distinguishing detail. Output exactly one line per tool, formatted:

<tool_name>: use when <condition>

No other text."""

out = {}
for cl in CL:
    specs = "\n".join(f"  {t['name']}: {t['description']} (arguments: {', '.join(t['params'])})"
                      for t in cl["tools"])
    body = {"messages": [{"role": "user", "content": ASK.format(specs=specs)}],
            "temperature": 0.0, "max_tokens": 400, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        txt = json.loads(r.read())["choices"][0]["message"]["content"].strip()
    keep = {}
    for line in txt.splitlines():
        line = line.strip().lstrip("-* ").strip()
        for t in cl["tools"]:
            if line.lower().startswith(t["name"].lower()) and ":" in line:
                keep[t["name"]] = line.split(":", 1)[1].strip()
    out[str(cl["cluster"])] = keep
    print(f"cluster {cl['cluster']}: {len(keep)}/{len(cl['tools'])} rules")
    for k, v in keep.items(): print(f"    {k}: {v[:110]}")
json.dump(out, io.open("sharpened.json", "w", encoding="utf-8"), indent=1)
