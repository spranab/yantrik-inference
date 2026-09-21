"""A hard routing benchmark: options that genuinely overlap.

The easy benchmark is saturated because its 24 tools do different jobs. The
failure we actually saw was read-vs-edit: two tools that both touch a file,
told apart only by what the request implies. xlam has many such pairs, so here
the options are a cluster of near-duplicate tools and nothing else. The gold
label is the tool the dataset generated the query from, and the query's
arguments are what distinguish them.
"""
import json, io, collections, glob, random, re

SRC = glob.glob("C:/Users/sync/.cache/huggingface/hub/datasets--NobodyExistsOnTheInternet--xlam-function-calling-60k/snapshots/*/xlam_function_calling_60k.json")[0]
MIN_CASES, SIM = 55, 0.30
STOP = set("the a an of on in to for from by based given with and or using specified certain list".split())

def words(s):
    return {w for w in re.split(r"[^a-z0-9]+", s.lower()) if w and w not in STOP and len(w) > 2}

d = json.load(io.open(SRC, encoding="utf-8"))
freq, descs, params, cases = collections.Counter(), collections.defaultdict(collections.Counter), {}, collections.defaultdict(list)
for r in d:
    tools = json.loads(r["tools"]) if isinstance(r["tools"], str) else r["tools"]
    ans   = json.loads(r["answers"]) if isinstance(r["answers"], str) else r["answers"]
    names = {a["name"] for a in ans}
    if len(names) != 1: continue
    g = next(iter(names)); byname = {t["name"]: t for t in tools}
    if g not in byname: continue
    dsc = byname[g].get("description", "").strip()
    freq[g] += 1; descs[g][dsc] += 1
    cases[g].append((r["query"].strip(), dsc, ans[0].get("arguments", {})))
    params.setdefault((g, dsc), list(byname[g].get("parameters", {}).keys()))

# tools with one consistent spec and enough cases
pool = {}
for n, c in freq.items():
    top, ntop = descs[n].most_common(1)[0]
    if ntop >= MIN_CASES and ntop / c >= 0.95:
        pool[n] = top
print(f"{len(pool)} tools with a consistent spec and >= {MIN_CASES} cases")

# cluster by name+description overlap
W = {n: words(n + " " + t) for n, t in pool.items()}
names = sorted(pool)
parent = {n: n for n in names}
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
pairs = 0
for i, a in enumerate(names):
    for b in names[i+1:]:
        j = len(W[a] & W[b]) / max(1, len(W[a] | W[b]))
        if j >= SIM:
            parent[find(a)] = find(b); pairs += 1
clusters = collections.defaultdict(list)
for n in names: clusters[find(n)].append(n)
clusters = [sorted(v) for v in clusters.values() if 2 <= len(v) <= 5]
clusters.sort(key=lambda c: -sum(freq[n] for n in c))
print(f"{pairs} overlapping pairs -> {len(clusters)} clusters of 2-5 tools\n")
for c in clusters[:14]:
    print("  " + " | ".join(f"{n}({freq[n]})" for n in c))
    for n in c: print(f"      {n:32s} {pool[n][:88]}")

rng = random.Random(20260921)
out = []
for ci, c in enumerate(clusters):
    if len(c) < 2: continue
    tools = [{"name": n, "description": pool[n], "params": params[(n, pool[n])]} for n in c]
    ev, dev, calib = [], [], []
    for n in c:
        qs = sorted({q for q, dsc, _ in cases[n] if dsc == pool[n]}); rng.shuffle(qs)
        dev   += [{"query": q, "gold": n} for q in qs[:8]]        # shown as examples
        calib += [{"query": q, "gold": n} for q in qs[8:16]]      # fits the option bias
        ev    += [{"query": q, "gold": n} for q in qs[16:16 + 30]]
    for x in (ev, dev, calib): rng.shuffle(x)
    out.append({"cluster": ci, "tools": tools, "dev": dev, "calib": calib, "eval": ev})

tot = sum(len(c["eval"]) for c in out)
io.open("bench_hard.json", "w", encoding="utf-8").write(json.dumps(out, indent=1))
print(f"\n{len(out)} clusters, {tot} eval cases, {sum(len(c['dev']) for c in out)} dev")
