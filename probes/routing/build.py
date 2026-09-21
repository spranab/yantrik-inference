"""Build a fixed-universe tool-routing benchmark from xlam-60k.

The endpoint caches a shared preamble, so the benchmark puts the tool list in
the preamble and only the user query in the record. That is the shape of an
agent router with a fixed tool set, which is what we are trying to improve.

Label noise matters here: if two tools in the universe do the same job, the
dataset's choice between them is arbitrary. So the universe is built greedily
from the most frequent tools, skipping any whose name+description overlaps one
already accepted.
"""
import json, io, collections, glob, random, re

SRC = glob.glob("C:/Users/sync/.cache/huggingface/hub/datasets--NobodyExistsOnTheInternet--xlam-function-calling-60k/snapshots/*/xlam_function_calling_60k.json")[0]
K, N_EVAL, N_DEV, JACCARD = 24, 40, 24, 0.22
STOP = set("the a an of on in to for from by based given with and or using calculate calculates compute computes get retrieve retrieves find finds return returns value values specified certain list number numbers".split())

def words(s):
    return {w for w in re.split(r"[^a-z0-9]+", s.lower()) if w and w not in STOP and len(w) > 2}

d = json.load(io.open(SRC, encoding="utf-8"))
freq, descs, params, cases = collections.Counter(), collections.defaultdict(collections.Counter), {}, collections.defaultdict(list)
for r in d:
    tools = json.loads(r["tools"]) if isinstance(r["tools"], str) else r["tools"]
    ans   = json.loads(r["answers"]) if isinstance(r["answers"], str) else r["answers"]
    names = {a["name"] for a in ans}
    if len(tools) < 2 or len(names) != 1:
        continue
    g = next(iter(names))
    byname = {t["name"]: t for t in tools}
    if g not in byname:
        continue
    dsc = byname[g].get("description", "").strip()
    freq[g] += 1
    descs[g][dsc] += 1
    cases[g].append((r["query"].strip(), dsc))
    params.setdefault((g, dsc), list(byname[g].get("parameters", {}).keys()))

universe, skipped, desc = [], [], {}
for n, c in freq.most_common():
    if len(universe) >= K:
        break
    top, ntop = descs[n].most_common(1)[0]
    if ntop < N_EVAL + N_DEV:
        continue
    if ntop / c < 0.95:                      # same name, several different tools
        skipped.append((n, f"{len(descs[n])} different descriptions, modal {ntop}/{c}")); continue
    desc[n] = top
    w = words(n + " " + desc[n])
    clash = next((u for u in universe
                  if len(w & words(u + " " + desc[u])) / max(1, len(w | words(u + " " + desc[u]))) > JACCARD), None)
    if clash:
        skipped.append((n, clash)); continue
    universe.append(n)

print(f"universe of {len(universe)} tools; skipped as too close to an accepted tool:")
for n, c in skipped[:12]:
    print(f"    {n:34s} ~ {c}")
print()
for n in universe:
    print(f"  {freq[n]:4d}  {n:34s} {desc[n][:78]}")

rng = random.Random(20260921)
dev, ev = [], []
for n in universe:
    qs = sorted({q for q, dsc in cases[n] if dsc == desc[n]}); rng.shuffle(qs)
    dev += [{"query": q, "gold": n} for q in qs[:N_DEV]]
    ev  += [{"query": q, "gold": n} for q in qs[N_DEV:N_DEV + N_EVAL]]
rng.shuffle(dev); rng.shuffle(ev)

out = {"universe": [{"name": n, "description": desc[n], "params": params[(n, desc[n])]} for n in universe],
       "dev": dev, "eval": ev}
io.open("bench.json", "w", encoding="utf-8").write(json.dumps(out, indent=1))
print(f"\n{len(dev)} dev + {len(ev)} eval cases, balanced, chance = {1/len(universe):.3f}")
