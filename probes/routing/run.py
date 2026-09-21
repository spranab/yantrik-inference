"""Routing accuracy on /v1/decide: how much does the cached preamble buy?

Every arm asks the same 24-way question about the same 960 queries. Only the
preamble changes, and the preamble is prefilled once and cached, so richer
arms cost nothing per request. Options are letters because 24 snake_case tool
names do not have distinct first tokens.
"""
import json, io, time, sys, statistics, urllib.request, string, math, collections

URL   = "http://127.0.0.1:8080/v1/decide"
B     = json.load(io.open("bench.json", encoding="utf-8"))
UNI   = B["universe"]
LET   = list(string.ascii_uppercase)[:len(UNI)]
QTEXT = "Which tool should be called to answer the request? Reply with its letter."

def post(body, tries=3):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except Exception as e:
            if i == tries - 1: raise
            time.sleep(2)

def preamble(kind, shots=0, dev=None):
    lines = ["TOOLS"]
    for l, t in zip(LET, UNI):
        row = f"  {l}. {t['name']}"
        if kind in ("desc", "params"):
            row += f" -- {t['description']}"
        if kind == "params" and t["params"]:
            row += f" (arguments: {', '.join(t['params'])})"
        lines.append(row)
    s = "\n".join(lines)
    if shots:
        idx = {t["name"]: l for l, t in zip(LET, UNI)}
        byc = collections.defaultdict(list)
        for c in dev: byc[c["gold"]].append(c)
        ex, k = [], shots // len(UNI)
        for t in UNI:
            for c in byc[t["name"]][:k]:
                ex.append(f"  request: {c['query']}\n  tool: {idx[c['gold']]}")
        s += "\n\nEXAMPLES\n" + "\n\n".join(ex)
    return s + "\n\nREQUEST\n"

def arm(name, pre, cases, bias=None):
    hits, probs, ms, cached, rows = 0, [], [], 0, []
    t0 = time.time()
    for i, c in enumerate(cases):
        r = post({"preamble": pre, "record": c["query"],
                  "questions": [{"q": QTEXT, "opts": LET}]})
        a = r["answers"][0]
        p = a["probabilities"]
        if bias:
            pick = max(p, key=lambda k: math.log(max(p[k], 1e-9)) + bias.get(k, 0.0))
        else:
            pick = a["answer"]
        gold = LET[[t["name"] for t in UNI].index(c["gold"])]
        hits += pick == gold
        probs.append(p); ms.append(r["seconds"] * 1000); cached += bool(r.get("preamble_cached"))
        rows.append({"q": c["query"], "gold": gold, "pick": pick, "conf": p.get(pick, 0.0), "p": p})
        if i % 120 == 119:
            print(f"    {name} {i+1}/{len(cases)} acc {hits/(i+1):.3f}", flush=True)
    n = len(cases); acc = hits / n
    se = math.sqrt(acc * (1 - acc) / n)
    print(f"  {name:22s} acc {acc:.3f} +-{1.96*se:.3f}  median {statistics.median(ms):.0f} ms  "
          f"cached {cached}/{n}  wall {time.time()-t0:.0f}s", flush=True)
    return {"arm": name, "acc": acc, "n": n, "ci": 1.96 * se, "median_ms": statistics.median(ms),
            "cached": cached, "rows": rows, "preamble_chars": len(pre)}

def mcnemar(a, b):
    """paired: did arm b fix more than it broke?"""
    fixed = sum(1 for x, y in zip(a["rows"], b["rows"]) if x["pick"] != x["gold"] and y["pick"] == y["gold"])
    broke = sum(1 for x, y in zip(a["rows"], b["rows"]) if x["pick"] == x["gold"] and y["pick"] != y["gold"])
    n = fixed + broke
    # two-sided exact binomial p under 50/50
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(fixed, broke) + 1)) / 2**n)
    return fixed, broke, p

if __name__ == "__main__":
    dev, ev = B["dev"], B["eval"]
    if len(sys.argv) > 1: ev = ev[:int(sys.argv[1])]
    print(f"{len(UNI)} tools, {len(ev)} eval cases, chance {1/len(UNI):.3f}\n")
    out = {}

    pres = {"names only":        preamble("names"),
            "+ descriptions":    preamble("desc"),
            "+ argument names":  preamble("params"),
            "+ 48 examples":     preamble("params", shots=48, dev=dev)}
    for k, pre in pres.items():
        print(f"  [{k}] preamble {len(pre)} chars")
        out[k] = arm(k, pre, ev)

    # calibration: the model's marginal over options on dev should be uniform,
    # since dev is balanced. fit one offset per option, apply at read time.
    best = max(out, key=lambda k: out[k]["acc"])
    print(f"\n  fitting option bias on {len(dev)} dev cases under [{best}]")
    d = arm("dev(" + best + ")", pres[best], dev)
    mean = {l: statistics.mean(r["p"].get(l, 0.0) for r in d["rows"]) for l in LET}
    bias = {l: -math.log(max(mean[l], 1e-6)) for l in LET}
    out["+ calibration"] = arm("+ calibration", pres[best], ev, bias=bias)
    out["_dev"] = d; out["_bias"] = bias; out["_best"] = best

    print()
    keys = [k for k in out if not k.startswith("_")]
    for a, b in zip(keys, keys[1:]):
        f, br, p = mcnemar(out[a], out[b])
        print(f"  {a:18s} -> {b:18s} fixed {f:3d} broke {br:3d}  p {p:.4f}")
    json.dump(out, io.open("results.json", "w"), indent=1)
