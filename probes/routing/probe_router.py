"""The 7-tool agent router: does a sharper preamble fix the one failure?

Ten cases cannot measure a difference -- that is what bench_hard.json is for.
This checks whether the specific read-vs-edit failure has a specification cause,
by varying two things the caller controls for free: the wording of the question,
and what the cached preamble says about where one tool ends and the next begins.
"""
import json, io, urllib.request, sys

URL   = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080") + "/v1/decide"
TOOLS = ("bash", "write", "edit", "read", "search", "answer", "agent")

FLAT = ("TOOLS\n"
        "  bash    run a shell command: check state, measure, ssh, start or stop things\n"
        "  write   create a new file that does not exist yet\n"
        "  edit    change an existing file\n"
        "  read    open a file to see what is in it\n"
        "  search  look something up on the public web\n"
        "  answer  reply directly, calling no tool\n"
        "  agent   hand a large open-ended job to a sub-agent\n\n"
        "REQUEST\n")

# each line says what to choose this tool *instead of*, which is the thing a
# one-line description never says
SHARP = ("TOOLS\n"
         "  bash    run a shell command: check state, measure, ssh, start or stop things.\n"
         "          Prefer it over read when the answer is a fact about the machine rather than the contents of a file.\n"
         "  write   create a file that does not exist yet. Prefer it over edit when there is nothing there to change.\n"
         "  edit    change a file when the request already states the change to make.\n"
         "          Prefer it over read: needing to look at the file first does not make this a read.\n"
         "  read    open a file to see what is in it, when looking is the whole request.\n"
         "  search  look something up on the public web. Prefer it over answer when the fact lives outside\n"
         "          this machine and may have changed since.\n"
         "  answer  reply directly, calling no tool. Prefer it when the request is about this conversation,\n"
         "          asks for an explanation, or asks for a judgement.\n"
         "  agent   hand a large open-ended job to a sub-agent. Prefer it over bash when the job is many\n"
         "          steps whose shape is not known yet.\n\n"
         "REQUEST\n")

SHOTS = ("\nEXAMPLES\n"
         "  request: does the cache survive a restart?                         tool: bash\n"
         "  request: start a new module for the retry logic                    tool: write\n"
         "  request: rename that variable to n_batch everywhere in the file    tool: edit\n"
         "  request: show me what the config sets for the port                 tool: read\n"
         "  request: which version of cuda does torch 2.9 need?                tool: search\n"
         "  request: why did that fail?                                        tool: answer\n"
         "  request: survey every backend and report which ones support this    tool: agent\n")

CASES = [("check if the service on port 8080 is up", "bash"),
         ("what did we decide about the cache earlier?", "answer"),
         ("add a --verbose flag to the CLI parser", "edit"),
         ("write a script that benchmarks this endpoint", "write"),
         ("what is the newest release of numpy?", "search"),
         ("open the handoff document and pick up where it stopped", "read"),
         ("go through every paper on this topic and summarise them", "agent"),
         ("is that a good idea?", "answer"),
         ("the deploy failed again, look at the logs", "bash"),
         ("explain that in simpler terms", "answer")]

Q_FIRST = "Which tool should be used first?"
Q_DO    = "Which tool should be called to carry out this request?"

VARIANTS = [("flat spec, 'used first'",      FLAT,  Q_FIRST),
            ("flat spec, 'carry out'",       FLAT,  Q_DO),
            ("sharp spec, 'carry out'",      SHARP, Q_DO),
            ("sharp spec + examples",        SHARP.replace("\nREQUEST\n", "") + SHOTS + "\nREQUEST\n", Q_DO)]

def post(body):
    r = urllib.request.Request(URL, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=300) as resp: return json.loads(resp.read())

print(f"  {len(CASES)} cases, {len(TOOLS)} tools. Ten cases cannot measure a difference; this shows where it moves.\n")
picks = {}
for name, pre, q in VARIANTS:
    ok, out = 0, []
    for text, gold in CASES:
        r = post({"preamble": pre, "record": text, "questions": [{"q": q, "opts": list(TOOLS)}]})
        a = r["answers"][0]
        ok += a["answer"] == gold
        out.append((a["answer"], a["confidence"]))
    picks[name] = out
    print(f"  {ok}/{len(CASES)}  {name}  (preamble {len(pre)} chars)")

print()
hdr = "  request".ljust(52) + "want     " + "".join(n.split(",")[0][:11].ljust(13) for n, _, _ in VARIANTS)
print(hdr)
for i, (text, gold) in enumerate(CASES):
    row = "  " + text[:48].ljust(50) + gold.ljust(9)
    for name, _, _ in VARIANTS:
        p, c = picks[name][i]
        row += f"{p}({c*100:.0f}%)".ljust(13) if p == gold else f"{p}({c*100:.0f}%)!".ljust(13)
    print(row)
