"""Generic resumable steering runner: jobs file -> generations -> Jev trait profiles.

python run_jobs.py --jobs out/hunt/screen_jobs.json --set SCREEN --max-tokens 80 --tag screen
Each job: {"id": str, "targets": [[layer, neuron, dose], ...], "meta": {...}}.
Generations append to <tag>.jsonl (one row per job x prompt); Jev profiles to <tag>.judged.jsonl.
Jobs already in <tag>.jsonl are skipped, so the runner can be killed and restarted.
"""
import argparse, json, threading, time
from concurrent.futures import ThreadPoolExecutor
from common import *
import prompts as P
import traits

ap = argparse.ArgumentParser()
ap.add_argument("--jobs", required=True)
ap.add_argument("--set", default="SCREEN")
ap.add_argument("--max-tokens", type=int, default=80)
ap.add_argument("--tag", required=True)
ap.add_argument("--deadline", type=float, default=0)
a = ap.parse_args()

prompts = getattr(P, a.set)
gen_path, jud_path = RUN / f"{a.tag}.jsonl", RUN / f"{a.tag}.judged.jsonl"
done = set()
if gen_path.exists():
    done = {json.loads(l)["id"] for l in gen_path.read_text().splitlines() if l.strip()}
judged = set()
if jud_path.exists():
    judged = {(json.loads(l)["id"], json.loads(l)["i"]) for l in jud_path.read_text().splitlines() if l.strip()}
lock = threading.Lock()
pool = ThreadPoolExecutor(8)


def judge_one(row):
    try:
        s = traits.score(row["prompt"], row["output"])
        rec = {"id": row["id"], "i": row["i"], "jev": s}
    except Exception as e:
        rec = {"id": row["id"], "i": row["i"], "error": repr(e)[:300]}
    with lock, open(jud_path, "a") as f:
        f.write(json.dumps(rec) + "\n")


model, tok = load_model()
calib = calibrate(model, tok, [p.strip() for p in (HERE.parent / "prompts.txt").read_text().splitlines()
                               if p.strip() and not p.startswith("#")])
# re-judge rows generated earlier whose judging never landed
if gen_path.exists():
    for l in gen_path.read_text().splitlines():
        r = json.loads(l)
        if (r["id"], r["i"]) not in judged:
            pool.submit(judge_one, r)

while True:
    jobs = json.loads(Path(a.jobs).read_text())
    todo = [j for j in jobs if j["id"] not in done]
    if not todo:
        break
    j = todo[0]
    if a.deadline and time.time() > a.deadline:
        say(a.tag, "deadline reached"); break
    t0 = time.time()
    with Steer(model, calib, [tuple(x) for x in j["targets"]]):
        outs = gen(model, tok, prompts, a.max_tokens)
    rows = [{"id": j["id"], "i": i, "targets": j["targets"], "meta": j.get("meta", {}), "prompt": p, "output": o}
            for i, (p, o) in enumerate(zip(prompts, outs))]
    with open(gen_path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for r in rows:
        pool.submit(judge_one, r)
    done.add(j["id"])
    say(a.tag, j["id"], f"{time.time()-t0:.1f}s", "|", outs[0][:90].replace("\n", " "))
pool.shutdown(wait=True)
say(a.tag, "finished")
