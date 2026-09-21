"""Is the letter indirection costing accuracy?

The engine scores an option by its first token, so two names that start with the
same token are rejected and the caller falls back to letters. Where the names
happen to be distinguishable, the same cases can be asked both ways, which says
whether a fix for prefix-sharing names (scoring the whole option) is worth
building.
"""
import json, io, urllib.request, statistics, math, string, sys

BASE  = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
CL    = json.load(io.open("bench_hard.json", encoding="utf-8"))
SHARP = json.load(io.open("sharpened.json", encoding="utf-8"))
QTEXT = "Which tool should be called to answer the request?"

def post(path, body):
    r = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=300) as resp: return json.loads(resp.read())

def first_tokens(opt):
    """every spelling the engine scores, reduced to its first token"""
    out = set()
    for v in (opt, " " + opt, opt[:1].upper() + opt[1:], " " + opt[:1].upper() + opt[1:], opt.upper()):
        t = post("/tokenize", {"content": v, "add_special": False})["tokens"]
        if t: out.add(t[0])
    return out

def preamble(cl, by_name):
    lets = string.ascii_uppercase[:len(cl["tools"])]
    lines = ["TOOLS"]
    for l, t in zip(lets, cl["tools"]):
        head = f"  {t['name']}" if by_name else f"  {l}. {t['name']}"
        lines.append(head + f" -- {t['description']} (arguments: {', '.join(t['params'])})")
        rule = SHARP.get(str(cl["cluster"]), {}).get(t["name"])
        if rule: lines.append(f"       choose it when {rule}")
    idx = {t["name"]: (t["name"] if by_name else l) for l, t in zip(lets, cl["tools"])}
    byc = {}
    for c in cl["dev"]: byc.setdefault(c["gold"], []).append(c)
    ex = [f"  request: {c['query']}\n  tool: {idx[c['gold']]}"
          for t in cl["tools"] for c in byc.get(t["name"], [])[:8]]
    return "\n".join(lines) + "\n\nEXAMPLES\n" + "\n\n".join(ex) + "\n\nREQUEST\n"

scorable = []
for cl in CL:
    names = [t["name"] for t in cl["tools"]]
    fts = [first_tokens(n) for n in names]
    clash = any(fts[i] & fts[j] for i in range(len(fts)) for j in range(i + 1, len(fts)))
    (scorable.append(cl) if not clash else None)
    print(f"  cluster {cl['cluster']:2d} {'names scorable  ' if not clash else 'first tokens clash'} "
          f"{' | '.join(names)[:64]}")

print(f"\n  {len(scorable)}/{len(CL)} clusters can be asked by name today\n")
res = {}
for by_name in (False, True):
    rows, ms = [], []
    for cl in scorable:
        lets = string.ascii_uppercase[:len(cl["tools"])]
        names = [t["name"] for t in cl["tools"]]
        opts = names if by_name else list(lets)
        pre = preamble(cl, by_name)
        for c in cl["eval"]:
            r = post("/v1/decide", {"preamble": pre, "record": c["query"],
                                    "questions": [{"q": QTEXT, "opts": opts}]})
            a = r["answers"][0]
            gold = c["gold"] if by_name else lets[names.index(c["gold"])]
            rows.append((a["answer"] == gold, a["confidence"]))
            ms.append(r["seconds"] * 1000)
    acc = sum(x for x, _ in rows) / len(rows)
    res["names" if by_name else "letters"] = rows
    print(f"  {'tool names' if by_name else 'letters   '}  acc {acc:.3f} "
          f"+-{1.96*math.sqrt(acc*(1-acc)/len(rows)):.3f}  n {len(rows)}  "
          f"mean confidence {statistics.mean(c for _, c in rows):.3f}  median {statistics.median(ms):.0f} ms")

a, b = res["letters"], res["names"]
f = sum(1 for x, y in zip(a, b) if not x[0] and y[0]); br = sum(1 for x, y in zip(a, b) if x[0] and not y[0])
n = f + br
p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(f, br) + 1)) / 2 ** n)
print(f"\n  names fixed {f}, broke {br}, p {p:.4f}")
