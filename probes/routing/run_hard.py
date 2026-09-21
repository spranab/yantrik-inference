"""Arm ladder on the confusable-cluster benchmark.

Each cluster is its own routing problem with its own preamble, so cases are
grouped by cluster and the preamble is cached within a cluster. Only the
preamble changes between arms; the questions and cases are identical, so the
comparison is paired.
"""
import json, io, time, sys, statistics, urllib.request, string, math, collections, re

URL   = "http://127.0.0.1:8080/v1/decide"
CL    = json.load(io.open("bench_hard.json", encoding="utf-8"))
SHARP = json.load(io.open("sharpened.json", encoding="utf-8"))
QTEXT = "Which tool should be called to answer the request? Reply with its letter."

def post(body, tries=3):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=180) as r: return json.loads(r.read())
        except Exception:
            if i == tries - 1: raise
            time.sleep(2)

def norm(s):
    return re.sub(r"[^a-z ]", "", s.lower().replace("computes", "calculates")).strip()

def preamble(cl, kind, shots=0):
    lets = string.ascii_uppercase[:len(cl["tools"])]
    lines = ["TOOLS"]
    for l, t in zip(lets, cl["tools"]):
        row = f"  {l}. {t['name']}"
        if kind in ("desc", "params", "shots", "sharp"): row += f" -- {t['description']}"
        if kind in ("params", "shots", "sharp") and t["params"]: row += f" (arguments: {', '.join(t['params'])})"
        lines.append(row)
        if kind == "sharp":
            rule = SHARP.get(str(cl["cluster"]), {}).get(t["name"])
            if rule: lines.append(f"       choose it when {rule}")
    s = "\n".join(lines)
    if kind == "sharp": shots = 8
    if shots:
        idx = {t["name"]: l for l, t in zip(lets, cl["tools"])}
        byc = collections.defaultdict(list)
        for c in cl["dev"]: byc[c["gold"]].append(c)
        ex = [f"  request: {c['query']}\n  tool: {idx[c['gold']]}"
              for t in cl["tools"] for c in byc[t["name"]][:shots]]
        s += "\n\nEXAMPLES\n" + "\n\n".join(ex)
    return s + "\n\nREQUEST\n"

def run_arm(name, kind, shots=0, bias=None, split="eval"):
    rows, ms, cached = [], [], 0
    t0 = time.time()
    for cl in CL:
        lets = string.ascii_uppercase[:len(cl["tools"])]
        names = [t["name"] for t in cl["tools"]]
        pre = preamble(cl, kind, shots)
        b = bias.get(cl["cluster"]) if bias else None
        for c in cl[split]:
            r = post({"preamble": pre, "record": c["query"],
                      "questions": [{"q": QTEXT, "opts": list(lets)}]})
            a = r["answers"][0]; p = a["probabilities"]
            pick = (max(p, key=lambda k: math.log(max(p[k], 1e-9)) + b.get(k, 0.0)) if b else a["answer"])
            gold = lets[names.index(c["gold"])]
            rows.append({"cluster": cl["cluster"], "gold": gold, "pick": pick,
                         "conf": p.get(pick, 0.0), "p": p})
            ms.append(r["seconds"] * 1000); cached += bool(r.get("preamble_cached"))
    n = len(rows); acc = sum(r["pick"] == r["gold"] for r in rows) / n
    se = math.sqrt(acc * (1 - acc) / n)
    print(f"  {name:20s} acc {acc:.3f} +-{1.96*se:.3f}  n {n}  median {statistics.median(ms):.0f} ms  "
          f"cached {cached}/{n}  wall {time.time()-t0:.0f}s", flush=True)
    return {"arm": name, "acc": acc, "n": n, "ci": 1.96 * se,
            "median_ms": statistics.median(ms), "cached": cached, "rows": rows}

def mcnemar(a, b):
    f = sum(1 for x, y in zip(a["rows"], b["rows"]) if x["pick"] != x["gold"] and y["pick"] == y["gold"])
    br = sum(1 for x, y in zip(a["rows"], b["rows"]) if x["pick"] == x["gold"] and y["pick"] != y["gold"])
    n = f + br
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(f, br) + 1)) / 2**n)
    return f, br, p

if __name__ == "__main__":
    only = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if only: CL = [dict(c, **{"eval": c["eval"][:only]}) for c in CL]
    # which clusters cannot be resolved at all: two tools with the same spec
    degen = set()
    for cl in CL:
        d = [norm(t["description"]) for t in cl["tools"]]
        if len(set(d)) < len(d): degen.add(cl["cluster"])
    print(f"{len(CL)} clusters, {sum(len(c['eval']) for c in CL)} eval cases; "
          f"{len(degen)} clusters contain two tools with the same description\n")

    arms = collections.OrderedDict()
    for label, kind, sh in [("names only", "names", 0), ("+ descriptions", "desc", 0),
                            ("+ argument names", "params", 0), ("+ 8 examples each", "shots", 8),
                            ("+ boundary rules", "sharp", 8)]:
        arms[label] = run_arm(label, kind, sh)

    print("")
    print("  fitting per-cluster option bias on held-out cases under [+ boundary rules]")
    dv = run_arm("calibration fit", "sharp", 8, split="calib")
    bias = {}
    for cl in CL:
        rs = [r for r in dv["rows"] if r["cluster"] == cl["cluster"]]
        lets = string.ascii_uppercase[:len(cl["tools"])]
        mean = {l: statistics.mean(r["p"].get(l, 0.0) for r in rs) for l in lets}
        bias[cl["cluster"]] = {l: -math.log(max(mean[l], 1e-6)) for l in lets}
    arms["+ calibration"] = run_arm("+ calibration", "sharp", 8, bias=bias)

    keys = list(arms)
    print()
    for a, b in zip(keys, keys[1:]):
        f, br, p = mcnemar(arms[a], arms[b])
        print(f"  {a:18s} -> {b:18s} fixed {f:3d} broke {br:3d}  p {p:.4f}")

    best = max(arms.values(), key=lambda a: a["acc"])
    print(f"\n  per cluster, best arm [{best['arm']}]:")
    for cl in CL:
        rs = [r for r in best["rows"] if r["cluster"] == cl["cluster"]]
        a = sum(r["pick"] == r["gold"] for r in rs) / len(rs)
        tag = "  <- same description" if cl["cluster"] in degen else ""
        print(f"    {a:.3f}  {' | '.join(t['name'] for t in cl['tools'])[:72]}{tag}")
    res = [r for r in best["rows"] if r["cluster"] not in degen]
    ra = sum(r["pick"] == r["gold"] for r in res) / len(res)
    print(f"\n  resolvable clusters only: {ra:.3f} on {len(res)} cases")

    print("\n  confidence gating on the best arm:")
    for g in (0.5, 0.7, 0.9, 0.97, 0.99):
        keep = [r for r in best["rows"] if r["conf"] >= g]
        if not keep: continue
        esc = [r for r in best["rows"] if r["conf"] < g]
        ka = sum(r["pick"] == r["gold"] for r in keep) / len(keep)
        ea = (sum(r["pick"] == r["gold"] for r in esc) / len(esc)) if esc else float("nan")
        print(f"    gate {g:.2f}: keep {len(keep)/len(best['rows']):.2f} of cases at {ka:.3f}; "
              f"escalate {len(esc)} where it would have been {ea:.3f}")
    json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in arms.items()},
              io.open("results_hard.json", "w"), indent=1)
